This is a model which is predict if an object is a Rock or a Mine. First the "main\Copy of sonar data.csv" is undergone some preprocessing(this includes separating label and features and train_test_split) then train, with the training dataset, and then test it with test data, and check for accuracy. Then you can do a test by runnning any random dataset.
Here to make it simple, we are only using the labaratory data of SONAR to predict. We are using logistic regression model, its good when there are only 2 labels(i.e Rock or Mine)
Start a separate python environment with pyhton: python -m venv ml_env
Then run the command: git clone https://github.com/adarsh-v-h/Rock-v-s-Mine-Detection.git
Now setup requirements, run: pip install -r requirements.txt
Next you have to download dataset from drive: https://drive.google.com/drive/folders/1NEs0rpFelfzSWAJ6y832EDpW9ImQH4QJ , there you will see a "Copy of Sonar data.csv" download that and move it into your prj_dir\main or prj_dir/main .
Open for any contirbution or suggestion.
