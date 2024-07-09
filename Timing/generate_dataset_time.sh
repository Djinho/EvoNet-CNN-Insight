# generate_dataset.sh script for binary classification and strong selection

date
source $1

# Loop over batch index
for (( INDEX=1; INDEX<=$NBATCH; INDEX++ ))
do
    FNAME=$DIRDATA/Simulations$INDEX
    echo $FNAME
    mkdir -p $FNAME

    SEL=$SELCOEFF

    # Loop over recent and ancient times
    for TIME in ${RECENT[@]}
    do
        java -jar $DIRMSMS -N $NREF -ms $NCHROMS $NREPL -t $THETA -r $RHO $LEN -Sp $SELPOS -SI $TIME 1 $FREQ -SAA $(($SEL*2)) -SAa $SEL -Saa 0 -Smark $DEMO -threads $NTHREADS | gzip > $FNAME/msms..$SEL..recent..$TIME..txt.gz
    done

    for TIME in ${ANCIENT[@]}
    do
        java -jar $DIRMSMS -N $NREF -ms $NCHROMS $NREPL -t $THETA -r $RHO $LEN -Sp $SELPOS -SI $TIME 1 $FREQ -SAA $(($SEL*2)) -SAa $SEL -Saa 0 -Smark $DEMO -threads $NTHREADS | gzip > $FNAME/msms..$SEL..ancient..$TIME..txt.gz
    done
done
date
