This is my first time trying machine learning algorithms, so there may be several mistakes or whole logic may be wrong. 

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

<h2>Linear Regression with multiple variables (slightly underfit)</h2>
<ul>
  <li>Considered <a href="https://github.com/bp2881/machinelearning-algo/blob/pytorch/RealEstate.csv">Real Estate dataset</a></li>
  <li>Considered <b>"X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores"</b> as input features and <b>"Y house price of unit area"</b> as output factor</li>
  <li>Current optimal line => y = -3.09x1 - 7.08x2 + 3.70x3 + 37.94</li>
  <li>learning rate - 0.01, iterations - 300</li>
  <li>Prediction cost on test cases - <b>86.03</b></li>
  <img src="./assets/mv_plot.png" alt="There's nothing to see here, or is there?" width="75%">
</ul>

<h2>TODO:</h2>
<ul>
  <li>LinearRegression_mv is currently biased, fix that, EDIT: residual plot shows a pattern</li>
  <li>Update README in next commit</li>
</ul>

<h2>RESOURCES:</h2>
<ul>
  <li><a href="https://www.coursera.org/specializations/machine-learning-introduction">Machine Learning Specialization</a> - by Andrew Ng</li>
  <li><a href="https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E">Machine Learning from scratch</a> - by Patrick Loeber</li>
</ul>
