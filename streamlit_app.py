import streamlit as st
from PIL import Image

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#####################
# Header
st.write('''
# Harshit Dave
''')

image = Image.open('dp-modified.png')
st.image(image, width=150)

st.markdown('## About', unsafe_allow_html=True)
st.info('''
- I am a third-year undergraduate student pursuing IPG-M.Tech degree at IIIT Gwalior. I am currently exploring the field of deep learning and Django development.
- I have a keen interest in Big Data and love to solve DSA based and statistical-based problems.
- So far, I have Python, C++, Numpy, Pandas, Scikit-Learn, ML algorithms, Data Visualization, CNN, SQL and SQLite under my belt. I am learning new things in deep learning, Big Data and AutoML.
''')

#####################
# Navigation

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #07354a;">
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href="#education">Education</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#work-experience">Work Experience</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#project">Project</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#achievments-extracurriculars">Achievments/Extracurriculars</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#social-media">Social Media</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

#####################
# Custom function for printing text
def txt(a, b):
  col1, col2 = st.columns([4,1])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def txt2(a, b):
  col1, col2 = st.columns([1,4])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)

def txt3(a, b):
  col1, col2 = st.columns([1,2])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def txt4(a, b, c):
  col1, col2, col3 = st.columns([1.5,2,2])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)
  with col3:
    st.markdown(c)

#####################
st.markdown('''
## Education
''')

txt('**Integrated B.Tech + M.Tech** (Information Technology), *Indian Institute of Information Technology and Management*, Gwalior',
'2019-present')
st.markdown('''
- CGPA: `8.16`
- Undergraduate Courses: Probability and statistics, Artificial Intelligence, Data Structures, Analysis of Algorithms, Database Management system and Computer networks.
''')

txt('**Higher Secondary** (Science), *Swaminarayan Academy*, Surat',
'2007-2019')
st.markdown('''
- 12th Percentage: `93%`
- 10th CGPA: `10`
''')

#####################
st.markdown('''
## Work Experience
''')

txt('**Data Science Intern**, iSmile Technologies, Chicago,US , (Remote)',
'Aug 2021- Nov 2021')
st.markdown('''
- Worked on real-time automatic number plate recognition (RTANPR) project to automatically detect the vehicle’s number plate and store it into the database.
- Also prepared the report of RTANPR project for the client.
- Scraped the jobs from Linked-In using web scraping tools to prepare the company’s dashboard. Also wrote a few technical blogs related to data science fields.
- Worked on the initial preparation of the dashboard for the company.
- Technologies Used: SSD MobileNet, yolov4, opencv, Tensorflow, Python, Selenium, Excel.
''')

st.markdown('''
## Projects
''')

txt('**Fruit Freshness Detection** ',
'Dec 2021 - Jan 2021')
st.markdown('''
- Used `Sequential model (3 layer)`, then I improved the model by adding `ResNet50v2` and `VGG16` using transfer learning as our baseline model to detect the freshness of the fruits.
- Trained the model using one of the mentioned models and predicted the freshness of the fruit from its image.
- Got the accuracy of `Sequential Model : 96.96%`, `ResNet50v2: 97.41%` and `VGG16: 91.4%`.
- I used `Fruit fresh and rotten for classification` dataset available on the kaggle.
- Github: "https://github.com/harshitkd/Fruit-Freshness-Prediction"
''')

txt('**Automatic Number Plate recognition** ',
'Aug - Nov 2021')
st.markdown('''
- Used `Tensorflow object detection model (TFOD)` to detect the number plate of the vehicle. In this project I got the `average precision = 0.971`, `average recall (AR) = 0.7` and `F1 score = 0.81`.
- After detecting the number plate, used `EasyOCR` to get the text from the detected license plate and stored it in .csv file.
- Created a Tensorboard for visual representation of loss metric, learning rate, number of steps, mAP and AR.
- Github: "https://github.com/harshitkd/Real-Time-Number-Plate-Recognition"
''')

txt('**Real-Time Twitter Sentiment Analysis** ',
'Dec 2020 - Feb 2020')
st.markdown('''
- Developed an unsupervised real-time system that detects and analyses the tweet’s behaviour and sentiment in general.
- Implemented web scraping unit `twint`, and created a machine learning model using `K-means clustering`.
- Prepared the dashboard using `Dash` and dash-bootstrap to visualize the analysis of the tweets.
- Accuracy of the model: `75.2%`.
- Github: "https://github.com/gautamanirudh/twitterdash"
''')

#####################
st.markdown('''
## Achievments/Extracurriculars
''')

txt('**Data Science Honors**, WorldQuant University',
'Apr 2021- Sep 2021')
st.markdown('''
- Got selected in the Applied Data Science fellowship program and achieved honours in both courses (Highest possible grades) that were `Scientific Computing and Python for Data Science` and `Machine Learning and Statistical Analysis`.
- Worked on 3 mini projects in first course that were based on the topics: Data structures, OOPs, Pandas and statistics. In another course there were 2 mini projects based on the topics: Regression, classification, model selection, NLP, dimension reduction and feature selection.
- First course's credential - "https://www.credly.com/badges/bf3fe9a2-0892-441b-a36b-15c1f115cb30"
- Second course's credential - "https://www.credly.com/badges/bf3fe9a2-0892-441b-a36b-15c1f115cb30/public_url"
''')

txt('**Virtual Experience Program participant**, KPMG',
'March 2021')
st.markdown('''
- Participated in the open-access KPMG virtual internship program.
- Tasks I completed were: Data Quality Assessment, Data insights and presentation
- Credential - "https://drive.google.com/file/d/1dxOtxoQrqeca8VR7Zn3t6b0kppr-XQcD/view"
''')


#####################
st.markdown('''
## Skills
''')
txt3('Programming', '`Python`, `C/C++`, `Julia`')
txt3('Data processing/wrangling', '`SQL`, `pandas`, `numpy`')
txt3('Data visualization', '`matplotlib`, `seaborn`, `plotly`, `Dash`')
txt3('Machine Learning', '`scikit-learn`, `clustering`, `regression`')
txt3('Deep Learning', '`TensorFlow`, `keras`, `CNN`, `OpenCV`')
txt3('Web development', '`Django`, `HTML`, `CSS`, `streamlit`')

#####################
st.markdown('''
## Social Media
''')
txt2('LinkedIn', 'https://www.linkedin.com/in/harshit-dave/')
txt2('Twitter', 'https://twitter.com/iamhdave')
txt2('GitHub', 'https://github.com/harshitkd')
txt2('Leetcode', 'https://leetcode.com/con_dor/')
txt2('Codeforces', 'https://codeforces.com/profile/Josh_Lin')
txt2('Kaggle', 'https://www.kaggle.com/iamhdave')
