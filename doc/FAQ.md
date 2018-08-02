# FAQ

1. **Why can't I find some samples in the resulting graph?**

    During the process of tmap, it will drop some samples which is lack of required number of neighbors. Usually, these samples is taken as outlier or noise which is need to be obsoleted. If you are worried about the number of samples retaining in the graph, you could use the function called `tmap.tda.utils.cover_ratio` to calculate the cover ratio. Usually, cover ratio higher than 80% is basic need for tmap result.

1. **There are too much pairs which is significant co-enriched found in the result of coenrich. How could I choose from them?**

    In the FGFP example, it could see that it do have too much pairs found in the result of coenrich. The scores or p-value of co-enrich pairs are calculated base on the graph. Significant coenrichment is defined as pairs of features that share similar distribution along with the graph by SAFE score.

    Currently, we don't have better way to filter it again and extract more significant pairs. We could use several graphs with different parameters to find significant pairs under individual situations. We could also use p-value of coenrich to rank all pairs and extract the top.

1. **Why would I choose tmap instead of other traditional ordination analysis?**

    blabla....

1. **What does the meaning of SAFE score?**

    blabla....
1. **blabla...**

    blabla....
