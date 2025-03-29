# LP-ATS (Link Prediction with AttrTran Sim Score)

This project is the python code implementation of this journal paper - ["A Novel Similarity Score for Link Prediction Approach Using Financial Transaction Networks and Firms’ Attribute"](https://ieeexplore.ieee.org/abstract/document/10937185)

- [1] A. Bose and B. Kim, "A Novel Similarity Score for Link Prediction Approach Using Financial Transaction Networks and Firms’ Attribute," in IEEE Access, vol. 13, pp. 52051-52068, 2025, doi: 10.1109/ACCESS.2025.3553795.


## Project Structure
The notebook [nb-method-sample.ipynb](nb-method-sample.ipynb) loads and prepares the sample data from [data-sample.csv](data-sample.csv), runs the LP-ATS method to compute similarity score and measures AUC Score.

### Data Description
We have generated sample data - [data-sample.csv](data-sample.csv) that contains edge list with following columns:
1. `DATE`: Transaction Date in YYYYMMDD format
2. `SOURCE_ID`: Source Firm ID (Seller Company)
3. `SOURCE_SIC_CODE`: Attribute of Source Firm ID
4. `TARGET_ID`: Target Fird ID (Buyer Company)
5. `TARGET_SIC_CODE`: Attribute of Target Firm ID

## Usage
### Environment Setup

The code was tested in MacBook Pro with Apple M1 Pro processor with Conda environment in VSCode.

1. Install Conda [[ref](https://pytorch.org/get-started/locally/#anaconda)]:
    ```
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    sh Miniconda3-latest-MacOSX-arm64.sh
    ```
2. Create Conda environment from the provided `environment.yml`:
    ```
    conda env create -f environment.yml
    ```
3. Use `LP-ATS` Conda environment in VSCode.

### Run with your own data

1. Update [data-sample.csv](data-sample.csv) as required.
2. Open [nb-method-sample.ipynb](nb-method-sample.ipynb) and run the cells for each step.

### Method Parameters

The LP-ATS Similarity Score is computed using the following python method:
```
m = 2 # neighbor distance
alpha = 0.001  # influence
gamma = 0.8 # tuning parameter

p_matrix = functions.compute_lp_ats_sim_matrix(data, m=m, alpha=alpha, gamma=gamma)
```
