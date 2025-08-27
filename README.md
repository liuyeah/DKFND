## DKFND

Source code for our paper: "Detect, Investigate, Judge and Determine: A Knowledge-guided Framework for Few-shot Fake News Detection" on ICDM2025

## Detect Module
extract keywords:
`python preprocess/keywords_detect.py`

keywords encode: 
`python preprocess/news_encode.py`

### Inside Investigate, Judge and Determine
`python model/dem_inv.py `

### Outside InvestigateJ and Judge
`python preprocess/google_search.py`

### Outside Determine
`python model/google_inv.py`

### Final Determine
`python model/determine.py`

