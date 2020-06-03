library(data.table)

eQTX_list <- list()
for(f in dir(pattern=".*chr[0-9]*.*.csv")){
  print(f)
  i <- gsub("\\D","",f)
  print(i)
  dataset <- ifelse(grepl("methy",f),"methyl","acetyl")
  print(dataset)
  dt <- fread(f)
  print(head(dt))
  dt$dataset <- dataset
  dt$chr <- i
  eQTX_list[[f]] <- dt
}
eQTX_df <- rbindlist(eQTX_list,idcol="chr")

write.csv(eQTX_df, file = "eQTX_1MB_results.csv", row.names = F, quote = F)
