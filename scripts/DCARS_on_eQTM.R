
require(tidyverse)
require(data.table)
require(RSpectra)
require(R.matlab)
methy_data <- data.table::fread("tar zxfO ~/ROSMAPmethylationWAgeSex.tar.gz", header = T)
tmp_data <- scale(t(as.matrix(methy_data[,2:ncol(methy_data)])))
decomp <- RSpectra::svds(tmp_data,10)
probe_names <- methy_data$V1
rownames(methy_data) <- probe_names
phen_data <- read.csv("~/ROSMAP_PHEN.csv", row.names = 1)
mask <- match(colnames(methy_data), rownames(phen_data))
phen_data <- phen_data[rownames(phen_data) %in% colnames(methy_data), ]
methy_data <- methy_data[,rownames(phen_data), with=FALSE]
methy_data <- resid(lm(t(as.matrix(methy_data)) ~ phen_data$msex))
exprs <- readMat("~/expressionNonAgeSexAdj.mat")
expression <- exprs$Res[[1]]
rownames(expression) <- unlist(exprs$Res[[2]])
colnames(expression) <- unlist(exprs$Res[[3]])
phen <- readMat("~/phenotype.mat")
phenotype <- phen$Pm[[1]]
rownames(phenotype) <- unlist(phen$Pm[[2]])
colnames(phenotype) <- unlist(phen$Pm[[3]])
# regress out covariates
decomp <- svds(expression,10)
expression <- expression - (decomp$u %*% diag(decomp$d) %*% t(decomp$v))
expression <- expression[match(rownames(phenotype), rownames(expression)),]
all(rownames(expression) == rownames(phenotype))
fit <- lm(as.matrix(expression) ~  phenotype[,"msex"])
expression <- resid(fit)

eQTX_manifest <- read.delim("/home/wcasazza/eQTX_manifest.txt", sep = " ")
expression <- expression[rownames(expression) %in% rownames(methy_data),]
#methy_data <- methy_data[match(rownames(methy_data),rownames(expression))]
all(rownames(expression) == rownames(methy_data))
all(rownames(expression) == rownames(phen_data))
express_ranks <- apply(t(expression[order(phen_data$age.death),]),1,rank)
methy_ranks <- apply(t(methy_data[order(phen_data$age.death),]),1,rank)
colnames(methy_ranks) <- probe_names
edges<- apply(eQTX_manifest,1,function(x) strsplit(x[3],","))
names(edges) <- eQTX_manifest$gene
edges <- as.matrix(bind_rows(edges,.id = "gene"))
all_data <- rbind(t(express_ranks),t(methy_ranks))

# Run DCARS
W <- DCARS::weightMatrix(ncol(SKCM_rank), type = "triangular", span = 0.5, plot = FALSE)
niter <- 1000
result <- DCARS::DCARSacrossNetwork(all_data,edges,extractPermutationTestStatistics = TRUE,W=W ,verbose=F, niter = niter)
saveRDS(result, "DCARS_network_permstats.RDS")

result <- DCARS::DCARSacrossNetwork(all_data,edges,extractTestStatisticOnly=TRUE, W=W,verbose=F, niter = niter)
saveRDS(result, "DCARS_network_teststats.RDS")

result <- DCARS::DCARSacrossNetwork(all_data,edges,extractWcorSequenceOnly = TRUE, W=W,verbose=F, niter = niter)
saveRDS(result, "DCARS_network_wcor.RDS")
