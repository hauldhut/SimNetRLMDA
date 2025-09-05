library(igraph)
library(ggraph)
library(ggplot2)
library(dplyr)
library(hash)
library(cowplot)   # used later for combining

GeneInfo_file = "/Users/hauldhut/Manuscripts/125GNN4DR/Data/EntrezGeneInfo_New.txt"
gene_df <- read.table(
  GeneInfo_file,
  sep = "\t",
  header = TRUE,
  quote = "",
  comment.char = "",
  fill = TRUE,
  stringsAsFactors = FALSE
)
dim(gene_df)

geneSymbol2ID_hash = hash()
geneID2Symbol_hash = hash()
for(i in 1:nrow(gene_df)){
  geneid = as.character(gene_df$EntrezID[i])
  genesymbol = as.character(gene_df$Symbol[i])
  geneSymbol2ID_hash[[genesymbol]] = geneid
  geneID2Symbol_hash[[geneid]] = paste0(genesymbol, " (",geneid,")")
}

length(geneSymbol2ID_hash)
length(geneID2Symbol_hash)
# example lookup
# geneID2Symbol_hash[["22941"]]

setwd("~/Manuscripts/123GNN4MDA/Code")

# ========= LOAD DATA ==========
Disease_Info <- read.delim("/Users/hauldhut/Manuscripts/100MHMDA/Data/Phenotype2Genes_Full.txt", header = TRUE)
MiRNA_Info   <- read.delim("/Users/hauldhut/Data/miRNA/MTI/miRNATarget_All.txt", header = TRUE)
# MiRNA_Info   <- read.delim("/Users/hauldhut/Data/miRNA/MTI/miRNATarget_All_BSv6.txt", header = TRUE)
# MiRNA_Info assumed columns: miRNA, GeneList, Source (miRWalk, TargetScan, miRTarBase)

# Clean gene lists
Disease_Info$GeneList <- gsub("[\r\n ]", "", Disease_Info$GeneList)
MiRNA_Info$GeneList   <- gsub("[\r\n ]", "", MiRNA_Info$GeneList)

# Create hash maps for labels
hDi <- hash()
for(i in 1:nrow(Disease_Info)){
  row = Disease_Info[i,]
  diid = row$MIMID
  diname1 = strsplit(row$Name,";")[[1]][1]
  diname2 = tolower(strsplit(diname1,",")[[1]][1])
  diname = paste0(diname2," (", substr(diid,4,9), ")")
  hDi[[diid]] = diname
}
# example: hDi[["MIM613436"]]

hMi <- hash()
for(i in 1:nrow(MiRNA_Info)){
  row = MiRNA_Info[i,]
  miid = row$miRNA
  hMi[[miid]] = miid   # use miRNA ID as label
}

# ========= BUILD GRAPH ==========
# minimal changes here; build graph same as original
build_disease_miRNA_gene_graph <- function(disease_id, mirna_id, only_shared_genes = FALSE){
  
  g <- graph.empty(directed = FALSE)
  hNodeID2Name = hash()
  
  # Add disease node
  di_id = paste0("MIM", disease_id)
  g <- add_vertices(g, 1, name = di_id, type = "Disease")
  hNodeID2Name[[di_id]] = hDi[[di_id]]
  
  # Add miRNA node
  g <- add_vertices(g, 1, name = mirna_id, type = "miRNA")
  hNodeID2Name[[mirna_id]] = hMi[[mirna_id]]
  
  # Disease genes
  diGenes <- unique(strsplit(Disease_Info[Disease_Info$MIMID==di_id, ]$GeneList, ",")[[1]])
  
  # miRNA genes (may come from multiple sources)
  miRNA_rows <- MiRNA_Info[MiRNA_Info$miRNA==mirna_id, ]
  
  # Add disease-gene edges
  for(gene in diGenes){
    # avoid duplicate vertex addition error: add only if not present
    if(!(gene %in% V(g)$name)){
      g <- add_vertices(g, 1, name = gene, type = "Gene")
      hNodeID2Name[[gene]] = geneID2Symbol_hash[[gene]]
    }
    g <- g + edge(di_id, gene, type = "Disease-Gene", weight = 1)
  }
  
  # Add miRNA-gene edges (optionally filter to only shared genes)
  for(i in 1:nrow(miRNA_rows)){
    row <- miRNA_rows[i, ]
    miGenes <- unique(strsplit(row$GeneList, ",")[[1]])
    src <- row$Source
    
    if(only_shared_genes){
      miGenes <- intersect(miGenes, diGenes)   # keep only shared
    }
    
    for(gene in miGenes){
      if(!(gene %in% V(g)$name)){
        g <- add_vertices(g, 1, name = gene, type = "Gene")
        hNodeID2Name[[gene]] = geneID2Symbol_hash[[gene]]
      }
      # set edge type to include source so edges differ
      g <- g + edge(mirna_id, gene, type = paste0("miRNA-Gene (", src,")"), weight = 1)
    }
  }
  
  # Replace node IDs with names (labels) for plotting
  V(g)$name <- sapply(V(g)$name, function(x) {
    # if name exists in hash mapping return mapping, else keep original
    if (has.key(x, hNodeID2Name)) hNodeID2Name[[x]] else x
  })
  
  return(g)
}


# ========= VISUALIZATION ==========
visualize_graph <- function(g, graphname){
  # Merge node type & shape into one mapping (node_types)
  V(g)$node_types <- V(g)$type
  
  # Define combined aesthetics for node_types
  node_colors <- c("Disease"="darkorange", "miRNA"="darkgreen", "Gene"="lightblue")
  node_shapes <- c("Disease"=15, "miRNA"=16, "Gene"=18)
  
  # Edge colors — still by source/type
  edge_colors <- c(
    "Disease-Gene" = "black",
    "miRNA-Gene (miRWalk)" = "red",
    "miRNA-Gene (TargetScan)" = "purple",
    "miRNA-Gene (miRTarBase)" = "brown",
    "Disease-miRNA" = "dodgerblue"
  )
  
  p <- ggraph(g, layout="fr") +
    geom_edge_parallel(aes(color = type), width = 0.9, alpha = 0.8, strength = 0.15) +
    geom_node_point(aes(color = node_types, shape = node_types), size = 10) +
    geom_node_text(aes(label = name), repel = TRUE, size = 6) +
    # unified node legend
    scale_color_manual(name = "Node type", values = node_colors, na.value = "grey50") +
    scale_shape_manual(name = "Node type", values = node_shapes) +
    # edge legend
    scale_edge_color_manual(name = "Edge type", values = edge_colors, na.value = "grey50") +
    theme_void() +
    theme(
      text = element_text(size = 20),
      legend.text = element_text(size = 20),
      legend.title = element_text(size = 20),
      plot.title = element_text(size = 20, face = "bold")
    )
  
  # Save
  saveRDS(p, paste0("./Temp/",graphname,"_Figure.rdata"))
  ggsave(paste0("./Temp/", graphname, "_Figure.pdf"), plot=p, width=7, height=7, dpi=300)
  return(p)
}


# ========= EXAMPLE USAGE ==========
# visualize association of Disease MIM114480 with hsa-miR-17 (only shared genes)
MIMID = "114480"
miRNA = "hsa-miR-17"
g <- build_disease_miRNA_gene_graph(MIMID, miRNA, only_shared_genes = TRUE)
p <- visualize_graph(g, paste0("MIM",MIMID,miRNA))
print(p)

# ========= LOOP THROUGH MDA PREDICTIONS (unchanged) ==========
MDA_file <- "../Prediction/DiseaseSimNet_OMIM_gat_d_512_e_100_miRNANetWSB_gat_d_512_e_100_Balanced_MLP_top_100000000_predictions.csv_group_top20.txt_AbstractCount_TableS3.txt"
MDA_df = read.delim(MDA_file, sep = "\t", header = TRUE)
dim(MDA_df)

df_Draw <- data.frame(
  OMIMID = character(),
  miRNA  = character(),
  stringsAsFactors = FALSE
)

for(i in 1:nrow(MDA_df)){
  MIMID = MDA_df$MIM[i]
  di_id = paste0("MIM", MIMID)
  miRNA = MDA_df$miRNA[i]
  
  diGenes <- unique(strsplit(Disease_Info[Disease_Info$MIMID==di_id, ]$GeneList, ",")[[1]])
  miGenes <- unique(strsplit(MiRNA_Info[MiRNA_Info$miRNA==miRNA, ]$GeneList, ",")[[1]])
  
  cat(di_id,"\t", miRNA,"\n")
  
  sharedGenes = intersect(diGenes, miGenes)
  if(length(sharedGenes)>0){
    print(sharedGenes)
    
    g <- build_disease_miRNA_gene_graph(MIMID, miRNA, only_shared_genes = TRUE)
    p <- visualize_graph(g, paste0("MIM",MIMID,"-",miRNA))
    print(p)
    
    row <- c(OMIMID = di_id, miRNA = miRNA)
    df_Draw[nrow(df_Draw) + 1, ] <- row
  }
}
dim(df_Draw)

# ========= Combine subfigures (unchanged) ==========
p = list()
legend_plot = NULL
maxSubFig = 8
nuSubFig = 0

for(i in 1:nrow(df_Draw)){
  row <- df_Draw[i,]
  nuSubFig <- nuSubFig + 1
  if(nuSubFig>maxSubFig) next
  
  OMIMID = row$OMIMID
  miRNA = row$miRNA
  fig = readRDS(paste0("./Temp/", OMIMID, "-", miRNA, "_Figure.rdata"))
  
  if (i==1) { legend_plot <- fig }
  
  fig = fig + theme(
    plot.margin = margin(20, 20, 20, 20),
    legend.position = "none"
  )
  
  p = append(p, list(fig))
}

# Extract legend and combine into grid
legend <- get_legend(legend_plot + theme(legend.position = "right"))
legend_cell <- ggdraw(legend) + theme(plot.margin = margin(20, 20, 20, 20))
p <- append(p, list(legend_cell))

combined_plot <- plot_grid(
  plotlist = p,
  labels = c("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", ""),
  label_size = 20,
  ncol = 3,
  nrow = 3,
  align = "hv"
)

ggsave("./Temp/Figure4.pdf", plot = combined_plot, width = 15, height = 15, dpi = 600)
