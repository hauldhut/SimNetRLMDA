Method = "MH123_Balanced"#/H1_Balanced/H3_Balanced/MH123_Balanced
gamma = 0.5

start_time <- Sys.time()

library(RandomWalkRestartMH)
library(igraph)

#need to install foreach and doParallel packages for this code to run
library(foreach)
library(doParallel)

setwd("~/Manuscripts/123GNN4MDA/Code")

miRNASimNet1 <- read.delim("../Data/miRNANetW.txt",header = FALSE)
miRNASimNet1.frame <- data.frame(miRNASimNet1[[1]], miRNASimNet1[[3]])
miRNASimNet1.g <- graph.data.frame(d = miRNASimNet1.frame, directed = FALSE)
miRNASimNet1.weight = miRNASimNet1[[2]]
E(miRNASimNet1.g)$weight <- miRNASimNet1.weight

miRNASimNet2 <- read.delim("../Data/miRNANetS.txt",header = FALSE)
miRNASimNet2.frame <- data.frame(miRNASimNet2[[1]], miRNASimNet2[[3]])
miRNASimNet2.g <- graph.data.frame(d = miRNASimNet2.frame, directed = FALSE)
miRNASimNet2.weight = miRNASimNet2[[2]]
E(miRNASimNet2.g)$weight <- miRNASimNet2.weight

miRNASimNet3 <- read.delim("../Data/miRNANetB.txt",header = FALSE)
miRNASimNet3.frame <- data.frame(miRNASimNet3[[1]], miRNASimNet3[[3]])
miRNASimNet3.g <- graph.data.frame(d = miRNASimNet3.frame, directed = FALSE)
miRNASimNet3.weight = miRNASimNet3[[2]]
E(miRNASimNet3.g)$weight <- miRNASimNet3.weight

if(Method == "MH123_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet1.g,miRNASimNet2.g,miRNASimNet3.g),Layers_Name = c("miRNASimNet1","miRNASimNet2","miRNASimNet3"))  
  tau1 = 1
  tau2 = 1
  tau3 = 1
  tau <- c(tau1, tau2, tau3)
}else if(Method == "MH12_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet1.g,miRNASimNet2.g),Layers_Name = c("miRNASimNet1","miRNASimNet2"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "MH13_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet1.g,miRNASimNet3.g),Layers_Name = c("miRNASimNet1","miRNASimNet3"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "MH23_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet2.g,miRNASimNet3.g),Layers_Name = c("miRNASimNet2","miRNASimNet3"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "H1_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet1.g),Layers_Name = c("miRNASimNet1"))
  tau <- c(1)
}else if(Method == "H2_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet2.g),Layers_Name = c("miRNASimNet2"))
  tau <- c(1)
}else if(Method == "H3_Balanced"){
  miRNA_MultiplexObject <- create.multiplex(list(miRNASimNet3.g),Layers_Name = c("miRNASimNet3"))
  tau <- c(1)
}

DiSimNet <- read.delim("../Data/DiseaseSimNet_OMIM.txt",header = FALSE)  


DiSimNet.frame <- data.frame(DiSimNet[[1]], DiSimNet[[3]])
DiSimNet.weight = DiSimNet[[2]]

DiSimNet.g <- graph.data.frame(d = DiSimNet.frame, directed = FALSE)
E(DiSimNet.g)$weight <- DiSimNet.weight

disease_MultiplexObject <- create.multiplex(list(DiSimNet.g),
                                         Layers_Name = c("DiSimNet"))

#Add miDiRelation
miDi.frame <- read.csv("../Data/Phenotype2miRNAs_HMDD.csv", header = TRUE)
miDi.frame <- miDi.frame[which(miDi.frame$miRNA %in% miRNA_MultiplexObject$Pool_of_Nodes),]
miDi.frame <- miDi.frame[which(miDi.frame$disease %in% disease_MultiplexObject$Pool_of_Nodes),]

#add func for RWR on multiplex-heter nw
do_something <- function(miRNA_MultiplexObject,disease_MultiplexObject,
                         miDiRelation,SeedmiRNA, seeddisease, prd_miRNAs) {
  
  #Create multiplex-heterosgenous nw
  miDiRelation_miRNA <- miDiRelation[which(miDiRelation$miRNA %in% miRNA_MultiplexObject$Pool_of_Nodes),]
  
  #Create multiplex-heterosgenous nw
  miRNA_disease_Net <- create.multiplexHet(miRNA_MultiplexObject, disease_MultiplexObject, 
                                          miDiRelation_miRNA)
  
  miRNA_disease_Net_TranMatrix <- compute.transition.matrix(miRNA_disease_Net)
  
  #compute 
  Ranking_Results <- Random.Walk.Restart.MultiplexHet(miRNA_disease_Net_TranMatrix,
                                                      miRNA_disease_Net,SeedmiRNA,
                                                      seeddisease, r = gamma)
  
  #create labels for ranking results
  tf = Ranking_Results$RWRMH_Multiplex1
  
  tf$labels <- ifelse(tf$NodeNames %in% prd_miRNAs, 1, 0)
  
  #Balanced: Select all nodes with label=1 and equal number of random nodes with label=0
  label_1_indices <- which(tf$labels == 1)
  label_0_indices <- which(tf$labels == 0)
  n_label_1 <- length(label_1_indices)
  if (n_label_1 > 0 && length(label_0_indices) >= n_label_1) {
    sampled_label_0_indices <- sample(label_0_indices, n_label_1)
    selected_indices <- c(label_1_indices, sampled_label_0_indices)
    tf <- tf[selected_indices, ]
  }
  
  # calculating AUC
  resultspred = prediction(tf$Score, tf$labels)
  
  pauc.perf = performance(resultspred, measure = "auc")
  return(list(pauc.perf@y.values[[1]],data.frame(Scores=tf$Score, Labels=tf$labels)))
}

#count miRNA for each disease
sub_sum <- aggregate(miRNA~disease, data=miDi.frame, FUN=function(x) c(count=length(x)))

#extract disease with only k or more miRNAs
k=5
sub_sum <- sub_sum[which(sub_sum$miRNA>=k),]
sub_sum$disease_no <- c(1:length(sub_sum$miRNA))

#extract miDi.frame with only disease from sub_sum
miDi.frame1 <- miDi.frame[which(miDi.frame$disease %in% sub_sum$disease),]
rownames(miDi.frame1) <- NULL #reset frame index

#func to assign k groups for each set of disease-miRNA, 
#as well as increment group no. for each group (k=3)
assign_group_no <- function(sub_sum,miDi.frame1,k) {
  
  #set an empty data frame for a new miDi.frame
  mylist.names <- c("disease","miRNA", "disease_no","group_no")
  miDi.frame2 <- sapply(mylist.names,function(x) NULL)
  
  for (j in 1:length(sub_sum$disease)) {
    
    count = sub_sum$miRNA[[j]]
    
    if(count<k) next
    
    set_no = floor(count/k)
    
    print(paste(k,count, set_no))
    
    group_vec = vector()
    for(gi  in 1:(k-1)){
      group_vec = c(group_vec,rep(gi,set_no))
    }
    group_vec = c(group_vec,rep(k,count-set_no*(k-1)))
    
    subset <- miDi.frame[which(miDi.frame$disease==sub_sum$disease[[j]]),]
    subset$disease_no <- rep(j,count)
    subset$group <- group_vec
    
    miDi.frame2 <- rbind(miDi.frame2,subset)
  }
  return(miDi.frame2)
}

#assign each group of n/k miRNA-disease with an group id
out <- assign_group_no(sub_sum,miDi.frame1,k)
out$group <- (out$disease_no-1)*k+out$group
miDi.frame2 <- out

#set an empty data frame for a new miDi.frame
auc_results <- sapply(c("disease","group","auc"),function(x) NULL)

for (i in 1:max(miDi.frame2$group)) {
  seeddisease = unique(miDi.frame2$disease[which(miDi.frame2$group==i)])
  auc_results$disease[i] <- seeddisease
  auc_results$group[i] <- i
}

#set up paralell processing (adjust the no_cores as per running system)
no_cores <- 6
cl <- makeCluster(no_cores)
registerDoParallel(cl)

#loop through
res <- foreach(i = 1:max(miDi.frame2$group), .combine = rbind) %dopar% {
  
  library(RandomWalkRestartMH)
  library(igraph)
  library(ROCR)
  
  prd_miRNAs = miDi.frame2$miRNA[which(miDi.frame2$group==i)]
  seeddisease = unique(miDi.frame2$disease[which(miDi.frame2$group==i)])
  
  disease_relation = miDi.frame2[which(miDi.frame2$disease==seeddisease),]
  SeedmiRNA = disease_relation$miRNA[-c(which(disease_relation$miRNA %in% prd_miRNAs))]
  
  # get bipartite graph without prd_miRNAs - disease linkages
  miDiRelation <- miDi.frame2[-with(miDi.frame2, which(miRNA %in% prd_miRNAs & disease %in% seeddisease)),c("miRNA","disease")]
  
  res <- do_something(miRNA_MultiplexObject,disease_MultiplexObject,
                      miDiRelation,SeedmiRNA, seeddisease, prd_miRNAs)
  res <- append(res, seeddisease)
}

dim(res)
stopCluster(cl)

df.res = data.frame(trial = c(1:nrow(res)),auc = unlist(res[,1]))
df.res.final = merge(df.res, miDi.frame2[c("disease","group")], by.x="trial", by.y="group", all.x = TRUE)
df.res.final = unique(df.res.final)

Result_byTrialFile = paste0("../Results/",Method,"_byTrial_",DiseaseSimNet,"_ROC_KFold",k,".csv")
cat(Result_byTrialFile,"\n")
write.csv(df.res.final,Result_byTrialFile, row.names = FALSE, quote = FALSE)

aucavgbyTrial = round(mean(df.res.final$auc),3)
aucavgbyTrial.sd = round(sd(df.res.final$auc),3)

res.final = NULL
for(i in 1:nrow(res)){
  res.final = rbind(res.final, res[i,2][[1]])
}

Result_by_Score_n_LabelFile = paste0("../Results/",Method,"_byTrial_",DiseaseSimNet,"_Score_n_Label_KFold",k,".csv")
cat(Result_by_Score_n_LabelFile,"\n")
write.csv(res.final,Result_by_Score_n_LabelFile, row.names = FALSE, quote = FALSE)

library(ROCR)
resultspred = prediction(res.final$Scores, res.final$Labels)
auc.perf = performance(resultspred, measure = "auc")
aucavgbyAll = round(auc.perf@y.values[[1]],3)

cat("Method=",Method,'DiseaseSimNet=',DiseaseSimNet,'aucavgbyAll=',aucavgbyAll,'aucavgbyTrial=',aucavgbyTrial,"(+-",aucavgbyTrial.sd,")\n")

end_time <- Sys.time()
timediff = end_time - start_time
print(timediff)
