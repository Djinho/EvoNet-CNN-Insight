# 1. Generate a specific metafile data
 
# download file https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
 
file_meta <- "integrated_call_samples_v3.20130502.ALL.panel"
 
# in this example I am interested in these population codes, change to yours!
populations <- c("YRI", "CEU", "CHB")
file_out <- "mysamples"
 
subsetMetadata <- function(file_meta, populations, file_out) {
        values <- read.table(file_meta, header=T)
        ind <- which(values$pop %in% populations)
        print(table(values$pop[ind]))
        # write meta
        write.table(cbind(values$sample[ind], values$pop[ind]), quote=F, row.names=F, col.names=T, file=paste(file_out,".meta",sep="",collapse=""))
        # write samples
        cat(values$sample[ind], sep=",", file=paste(file_out,".txt",sep="",collapse=""))
}
 
subsetMetadata(file_meta, populations, file_out)
 
# this command will generate two files, mysamples.txt and mysamples.meta which will be used later on
 
# -----------------------------------# 6.2 genetic diversity / pairwise nucleotide differences (pi)
 
# 2. Gather data
 
# register and login at ensembl.org
# go to https://www.ensembl.org/Homo_sapiens/Tools/DataSlicer
# fill in the form
# format: VCF
# region lookup: e.g. MEFV is 16:3242027-3256633
# genotype URL: e.g. for chrom16 https://ftp.ensembl.org/pub/data_files/homo_sapiens/GRCh38/variation_genotype/ALL.chr16_GRCh38.genotypes.20170504.vcf.gz
# filters: by individuals
# copy and paste mysamples.txt file!
# click run
# download file once done
 
# ----------------------------------------------------------------------------
