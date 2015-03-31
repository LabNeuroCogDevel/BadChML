bchfile <- Sys.glob('/data/Luna1/MultiModal/Clock/*/MEG/*bad*txt')
dfl <- lapply(bchfile, read.table,sep="#", comment.char="")
for (f in bchfile) {
  dfl <- read.table(f,sep="#", comment.char="")
}
