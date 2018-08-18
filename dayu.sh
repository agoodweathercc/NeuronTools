#!/bin/bash
for graph in  'reddit_12K';
do
    for  functiontype in 'ricci' 'ricci_edge' 'deg' 'hop' ;
    do
        for testcheck in 'True';

    do
        #echo $testcheck
        cd
        ~/anaconda2/bin/python trivial_kernel.py $graph $functiontype $testcheck 'heavey'
    done
done
done
