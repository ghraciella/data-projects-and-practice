# Data Engineering Projects  


### This contains my data engineering focused projects. 

<p align="center">

<div style="text-align:center"><img src="https://www.interviewbit.com/blog/wp-content/uploads/2021/08/Data-Engineer-2-800x391.jpg" /></div>


</p>




<br>

---
---

## <span style="color:grey"> Overview </span>
---


![de_workflow](https://content.altexsoft.com/media/2020/08/data-processing-within-etl-pipeline-and-warehouse.png.webp)


### __Skills and Tasks__ :

* Data Ingestion - Webscrapping
* ETL (Extract, Transform and Load) development skills
* Cloud computing
* Database, data pipelines and systems design and maintenance
* Data systems optimization
* Automation and scripting
* Design testing and debugging


### __Software Architecture and Frameworks__ 

* Python
* Java
* SQL
* Scala (Apache spark)
* Hive

<br>


---
---
## <span style="color:grey"> Project Directory Links </span>

---

<br>

* Projects
  - TBA
  - TBA


<br>



---
---

## <span style="color:grey"> Project Folder Structure Template </span>
---

```

|
|__ project/        : data engineering project folder
    |
    |
    |__ data/           : folder containing raw and processed data files
    |   |__ raw/        : unprocessed data files from data sources
    |   |__ processed/  : usable data files for data modelling
    |
    |
    |__ src/            : folder containing main source codes scripts
    |   |__ data/
    |   |   |__ data_loader.py : contains script for loading data files
    |   |   |__ data_preprocessor.py : contains script for data preprocessing
    |   |___|__ data_etl.py : contains script for executing data etl process
    |   |
    |   |
    |   |__ETL/            : folder containing scripts for the ETL process
    |   |   |__ extract/    : contains files for data extraction
    |   |   |__ transform/  : contains files for data transformation/wrangling 
    |   |___|__ load/       : contains files for data loading
    |   |
    |   |
    |   |__ pipelines/      : folder containing data pipeline execution scripts
    |   |
    |   |__ error_handler/   : folder containing files for unit tests etc
    |   |   |__ loader_test.py
    |   |   |__ etl_test.py
    |   |   |__ pipeline_test.py 
    |   |___|__ etc
    |   
    |
    |__ requirements.txt   : dependency management file which contains the required libraries/packages in their original versions used for project development
    |
    |__ .gitignore     : set rule which state files for Git to not track
    |
    |__ README.md      : main project documentation file


```


