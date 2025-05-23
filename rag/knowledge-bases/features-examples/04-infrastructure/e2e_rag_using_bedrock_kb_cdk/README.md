
# Deploy e2e RAG solution (using Amazon Bedrock Knowledge Bases) via CDK

Ref Blog: https://aws.amazon.com/blogs/machine-learning/build-an-end-to-end-rag-solution-using-knowledge-bases-for-amazon-bedrock-and-the-aws-cdk/

<mark>By NO means this deployment is production-ready deployment. Please adjust the IAM polies and permissions as per your organization policy)</mark>

This is a complete setup for automatic deployment of end-to-end RAG workflow using Amazon Bedrock Knowledge Bases. 
Following resources will get created and deployed:
- IAM role
- Open Search Serverless Collection and Index OR an Aurora PostgreSQL Provisioned Cluster as a vector store
- Set up Data Source (DS) and Knowledge Base (KB)

## Deployment steps

```
    -  git clone https://github.com/aws-samples/amazon-bedrock-samples.git
    
reinvent-kb-features
    -  cd knowledge-bases/features-examples/04-infrastructure/e2e_rag_using_bedrock_kb_cdk
main

```
This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually. 

__NOTE:__ *This project assumes you have python3.8 installed.
If you want to use a later version, you may have to make changes to the dependency versions
in requirements.txt.*



To manually create a virtualenv on MacOS and Linux:

```
python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
.venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
pip install -r requirements.txt
```
Upgrade aws-cdk-lib 
```
pip install --upgrade aws-cdk-lib
```
### Pre-requisites: 
- A S3 bucket set up with your documents in a supported format (.txt, .md, .html, .doc/docx, .csv, .xls/.xlsx, .pdf).

- Another S3 bucket set up for multimodal storage destination if building multi-modal RAG

### IMPORTANT : Update Config file 
**Open `config.py` and adjust the below parameters as per your application configuration**:
- ACCOUNT_ID
- ACCOUNT_REGION
- RAG_PROJ_NAME
- CHUNKING_STRATEGY
- MAX_TOKENS
- OVERLAP_PERCENTAGE
- S3_BUCKET_NAME
- VECTOR_STORE_TYPE (Ensure you select either 'OSS' for an OpenSearch Serverless or 'Aurora' for an Aurora PostgreSQL vector store)
- MULTI_MODAL (Set to true if building multi-modal RAG)
- PARSING_STRATEGY - Choose Parsing Strategy  'BEDROCK_DATA_AUTOMATION' or'BEDROCK_FOUNDATION_MODEL (if MULTI_MODAL is set to true)
- MM_STORAGE_S3 - S3 bucket for multimodal storage destination (if MULTI_MODAL is set to true). Make sure this bucket is in same region as of your data bucket.


**Save it!**

### PREPARATION: Bootstrap and synthesize the CDK project

At this point you can now synthesize the CloudFormation template. 

When deploying for the *first time*, run:
```
cdk bootstrap
```


```
cdk synth
```

### DEPLOYMENT: Deploy all stacks

This deployment contains multiple stacks (IAM role, vector store, and knowledge base). To deploy all the stacks in the proper sequence, use the 'cdk deploy --all' command.

```
cdk deploy --all
```

### DELETION: Destroying the deployed infrastructure

To Destroy the stack(s)

```
cdk destroy --all
```

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
