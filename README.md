This contains the full data and code used for the follow up case study on ATC in single vs. double blind reviewer policy. 

To recreate the full results, files should be run in the following order:

1. 011_scraper.rmd : Pulls arXiv matches for ICLR 2018 submissions
2. 012_data_pre_df_make : Forms data sets (all_2018, found_2018, all_2017_2018, all_2017_found_2018)
3. 013_init_analysis.rmd : Analyzes reviewer score distributions / etc. of data sets above
4. 020_embeddings.ipynb : Creates bow and doc2vec embeddings.
5. 030_atc_estimates.ipynb : Builds upon Zhang et al. 2023 code (largely copied) to find ATC results for each data set and embedding combination. Re-tunes hyperparameters to maintain causal overlap.
6. 040_atc_results.rmd : Visualizes atc results found above.
