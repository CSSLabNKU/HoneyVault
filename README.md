# How to Design Secure Honey Vault Schemes

This is the artifact for our ACM CCS 2025 paper: "How to Design Secure Honey Vault Schemes".

Authors: Zhenduo Hou, Tingwei Fan, Fei Duan and Ding Wang.

In this paper, we demonstrate how to design secure honey vault schemes in a principled approach. We first identify three major types of vulnerabilities, and propose three critical design criteria based on rigorous theories, with each aiming to address one type of vulnerability. To meet these key criteria, we propose VaultGuard with an innovative NLE and HE-Adaptive to honey-encrypt a user’s real vault and the adaptive PPM, respectively. Our NLE eliminates the first and second vulnerabilities, while HE-Adaptive addresses the third. We also provide a proof-of-concept implementation of our VaultGuard as a password manager extension installed on the browser. We realize VaultGuard-NLE with HE-Adaptive using JavaScript ES2023, and Web Crypto API (see https://bit.ly/4m8UkRo) for cryptographic executions. We use AES in CTR mode with PBKDF2-SHA256 to execute symmetric password-based encryption (PBE). Our implementation supports functionalities commonly found in ordinary password vaults, including registration, password modification, account deletion, and login.

## Create and Activate a Python Environment

Create and activate the Python environment in the folder containing requirements.txt files and install the corresponding dependencies.

### Using Python venv

**For Linux/MacOS:**
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**For Windows:**
```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

### Using Conda

**For All platforms:**

```bash
conda create -n honeyvault python=3.12
conda activate honeyvault
pip install -r requirements.txt
```

## Usage

This guide provides step-by-step instructions to reproduce the core experimental results presented in Figure 6 of the paper. 

1. **Generate the vault list**: Generate a vault list of a specified size, where the first one is the real vault (from the Pastebin dataset) and the remaining vaults are decoy vaults generated via specific NLE;
2. **Perform the distinguishing attacks**: Perform the three types of attack methods summarized in Table 1 of the paper against the vault list generated in step (1);
3. **Plot the experimental results**: Reproduce the core experimental results shown in Figure 6 of the paper based on the attack results obtained in step (2).

**NOTE**: Considering that the time cost of steps 1 and 2  is relatively high, it is recommended to set ```num=100``` for reproducing the experimental results.

### NoCrack

1. Navigate to the NoCrack directory:
    ```bash
    cd NoCrack
    ```
2. Generate the vault list with a size of ```num=1000```:
    ```bash
    python generate_vaults.py --num 1000 
    ```
    **NOTE**: Considering that the time cost of steps 1 and 2  is relatively high, it is recommended to set ```num=100``` for reproducing the experimental results.

3. Perform the distinguishing attacks:

    You can use the following command to perform all attacks at once. The default parameter ```num``` is set as 1000. You need to modify ```num``` in ```attacks.sh``` or ```attacks.bat``` according to the size of the vault list you specified. Of course, you can also copy each attack command in ```attacks.sh``` or ```attacks.bat``` and run them separately.

    **For Linux/MacOS:**
    ```bash
    chmod +x attacks.sh # Run only once
    ./attacks.sh
    ```

    **For Windows:**
    ```bash
    ./attacks.bat
    ```
4. Plot the experimental results of ```scheme=nocrack```:
    ```bash
    python plot.py --scheme nocrack --num 1000
    ```
    You can also use ```scheme=both``` to plot the experimental results of the NoCrack and VaultGuard (provided that the steps to distinguishing attacks are performed on both NoCrack and VaultGuard respectively).


### VaultGuard

1. Navigate to the VaultGuard directory:
    ```bash
    cd VaultGuard
    ```
2. Generate the vault list with a size of ```num=1000```:
    ```bash
    python generate_vaults.py --num 1000 
    ```
    **NOTE**: Considering that the time cost of steps 1 and 2  is relatively high, it is recommended to set ```num=100``` for reproducing the experimental results.
3. Perform the distinguishing attacks:

    You can run all attacks simultaneously using the provided script. By default, the ```num``` parameter in the script is set to 1000. Please adjust this value in either ```attacks.sh``` (Linux/Mac) or ```attacks.bat``` (Windows) to match the size of vault list.

    **Note**: Before running these scripts, you must first perform the following commands in a **Windows environment**, Before performing the ```run_vaultguard.bat```, you need to complete the commands in the ```run_vaultguard.bat``` according to the files under the ```vaultguard_config``` folder.

    ```bash
    python convert2targuess.py --num 1000
    cd targuess/targuess2_guess
    python config.py
    cd src
    .\run_vaultguard.bat
    ```
    
    > **Troubleshooting**: If the terminal crashes immediately:
    >
    > - Run individual commands like: `.\targuess2_guess.exe -c ..\vaultguard_config\sweetvaults_1.txt_config.ini`
    > - If prompted for missing DLL files, download them from [dll-files.com](https://www.dll-files.com/) and place in the `src` folder
    
    This command generates the required password probability files in the targuess directory.
    
    After completing the above steps, you can run the following command to perform distinguishing attacks. For Linux/MacOS, you also need to copy the files generated in folder ```targess/targuess2_guess/vaultguard_results``` to the same location in your Linux/MacOS environment.


    **For Linux/MacOS:** 
    ```bash
    chmod +x attacks.sh # Run only once
    ./attacks.sh 
    ```
    
    **For Windows:**
    ```bash
    ./attacks.bat
    ```
4. Plot the experimental results of ```scheme=vaultguard```:
    ```bash
    python plot.py --scheme vaultguard --num 1000
    ```
    You can also use ```scheme=both``` to plot the experimental results of the NoCrack and VaultGuard (provided that the steps to distinguishing attacks are performed on both NoCrack and VaultGuard respectively).

## 3. Chrome extension for our VaultGuard

Chrome extension for VaultGuard is implemented in the VaultGuard_ex.zip file. Please follow the steps below to complete the deployment process:

1. Unzip the VaultGuard_ex.zip file
2. Open Chrome and type 'chrome://extensions/' in the Chrome address bar
3. Enable "Developer Mode"
4. Click on "Load unpacked"
5. Select the project directory containing ```manifest.json``` file
6. The extension installation is completed

After successful installation, Chrome will pop up the following message: "Model uploaded successfully! VaultGuard is ready to use."

## 4. Academic Data Use Disclaimer for Sensitive Password Dataset

**CONFIDENTIALITY NOTICE**  
This dataset containing real-world password information is provided strictly for the purpose of artifact evaluation associated with academic research. By accessing this data, you explicitly agree to the following terms:

### Usage Restrictions
You are granted access to this sensitive dataset solely for the designated artifact evaluation process and must not use, reproduce, or analyze the data for any other purpose, including but not limited to additional research projects, commercial applications, or personal experimentation.  You are responsible for all activities conducted under your credentials and must implement reasonable safeguards to prevent unauthorized access or misuse of the data. 

### Security Requirements
You must implement appropriate administrative, technical, and physical safeguards to protect this sensitive information from any unauthorized use, access, or disclosure.  All data must be stored in an encrypted format with strong access controls, and access must be limited strictly to personnel directly involved in the approved evaluation. 

### Non-Disclosure Obligations
You agree not to disclose, share, or disseminate any portion of this dataset, including derived information that could potentially identify individuals or compromise security.  This includes refraining from publishing or presenting specific password examples or patterns that could be traced back to the original dataset.

### Data Handling Protocol
Upon completion of the artifact evaluation, all copies of the dataset must be securely destroyed.  Sensitive data must never be stored in non-production environments or transferred outside the approved evaluation context. 

### Compliance and Consequences
Failure to comply with these terms constitutes a breach of this agreement and may result in immediate revocation of access privileges, institutional sanctions, and potential legal action in accordance with applicable data protection legislation. 

By proceeding with access to this dataset, you acknowledge that you have read, understood, and agreed to be bound by these terms for the protection of sensitive user information.