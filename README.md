This is my first time trying machine learning algorithms, so there may be several mistakes or whole logic may be wrong. 

In the newer commits/versions, google colab has been used instead of local computer via VS Code extension. The boilerplate for it is present in every file, if not needed ignore it.

<h2>About Data</h2>
<ol>
  <li><a href="https://github.com/bp2881/machinelearning-algo/blob/pytorch/RealEstate.csv">Real Estate dataset</a></li>
  <ul>
    <li>rows - 411, columns - 7</li>
    <li>trainset - 80%, testset - 20%</li>
    <li>Used features have been normalized using z-score in dataset</li>
  </ul>
  <li><a href="https://github.com/bp2881/machinelearning-algo/blob/pytorch/cancer.csv">Cancer dataset</a></li>
  <ul>
    <li>rows - 553, columns - 32</li>
    <li>trainset - 80%, testset - 20%</li>
  </ul>
</ol>

<h2>Linear Regression with one variable (underfit)</h2>
<ul>
  <li>Considered <a href="https://github.com/bp2881/machinelearning-algo/blob/pytorch/RealEstate.csv">Real Estate dataset</a></li>
  <li>Considered <b>"X3 distance to the nearest MRT station"</b> as input feature and <b>"Y house price"</b> as output factor</li>
  <li>Current optimal line => y = -9.43x + 37.88</li>
  <li>learning rate - 0.01, iterations - 300</li>
  <li>Prediction cost on test cases - <b>102.47</b></li>
  <img src="./assets/ov_plot.png" alt="There's nothing to see here, or is there?" width="75%">
</ul>

<h2>Linear Regression with multiple variables (underfit)</h2>
<ul>
  <li>Considered <a href="https://github.com/bp2881/machinelearning-algo/blob/pytorch/RealEstate.csv">Real Estate dataset</a></li>
  <li>Considered <b>"X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores"</b> as input features and <b>"Y house price of unit area"</b> as output factor</li>
  <li>Current optimal line => y = -3.09x1 - 7.08x2 + 3.70x3 + 37.94</li>
  <li>learning rate - 0.01, iterations - 300</li>
  <li>Prediction cost on test cases - <b>86.03</b></li>
  <img src="./assets/mv_plot.png" alt="There's nothing to see here, or is there?" width="75%">
</ul>

<h2>Logistic Regression (best fit)</h2>
<ul>
  <li>Considered <a href="https://github.com/bp2881/machinelearning-algo/blob/pytorch/diabetes_prediction_dataset.csv">Diabetes Prediction Dataset</a></li>
  <li>Considered <b>"age"</b>, <b>"bmi"</b>, <b>"HbA1c_level"</b>, <b>"blood_glucose_level"</b> as input features and <b>"diabetes"</b> as output factor</li>
  <li>Used Sigmoid Activation Function</li>
  <li>Current optimal curve => z = 0.79x1 + 0.46x2 + 1.51x3 + 1.06x4 - 3.83</li>
  <li>learning rate - 0.01, iterations - 10k</li>
  <li>Testing Loss: <b>0.1265</b>, Accuracy: <b>95.77</b></li>
  <img src="./assets/log_plot.png" alt="There's nothing to see here, or is there?" width="75%">
  <br>
  
  <i><b>The PDP does not necessarily span the full probability range because other features are fixed at their mean values, limiting the maximum achievable logit.</b></i>
</ul>

<h2>ABOUT MY APPROACH:</h2>
<p>Due to me still being in the learning phase, my approach will change constantly and sometimes drastically.

<h4>Current Approach:</h4>
<ol>
  <li>Data Cleaning & Analysis:
  <ul>
    <li>Find Outliers, Missing values, etc. and correct them.</li>
    <li>Learn more about data (finding correlation, auto correlation, Linearity, etc.).</li>
    <li>After getting satisfied no. of observations, I normalize the data (Z-score).</li>
  </ul>
  </li>
  <li> Prediction:
  <ul>
    <li>Starting with Simple Linear Regression (if conditions satisfied while analysis) and then step up slowly if it's no good.</li>
    <li>Split into Train and Test Data (80:20) and start prediction using Gradient Descent method.</li>
    <li>If any anamoly is found then switch up to using Cross Validation. Now Train, Test and Cross Validation Data is split using 60:20:20 ratio.</li>
  </ul>
  </li>
  <li>Repeating Steps Until satisfied:
  <ul>  
    <li>This is a loop. Have to constantly find better way to predict it using limited resources and making proper tradeoff b/w prediction and iteration.</li>
  </ul>
  </li>
</ol>
</p>

<h2>TODO:</h2>
<ul>
  <li>Next Update: For Realestate Data, Linear models underfit hence have to use Polynomial Regression. </li>
</ul>

<h2>RESOURCES:</h2>
<ul>
  <li><a href="https://www.coursera.org/specializations/machine-learning-introduction">Machine Learning Specialization</a> - by Andrew Ng</li>
  <li><a href="https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E">Machine Learning from scratch</a> - by Patrick Loeber</li>
</ul>
