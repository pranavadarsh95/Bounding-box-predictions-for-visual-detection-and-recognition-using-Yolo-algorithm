 Bounding-box-predictions-for-visual-detection-and-recognition-using-Yolo-algorithm<br/>
 anaconda python 3.5<br/>
 code source hint:-kaggle<br/>
 data set:-kaggle<br/>
# Visual Detection
It is the process of finding real-world object instances like car, bike, TV,
flowers, and humans in still images or Videos. It allows for the recognition, localization, and
detection of multiple objects within an image which provides us with a much better
understanding of an image as a whole. It is commonly used in applications such as image
retrieval, security, surveillance, and advanced driver assistance systems (ADAS).
# Bounding Box
The bounding box is a rectangle drawn on the  image  which  tightly fits the visualintheimage. A bounding box exists for every instance of every visualin the image. For the box,4 numbers (center x, center y, width,height) are predicted. This canbetrained usingadistance measure between predicted and ground truth bounding box. The distance measureis a jaccard distance which computes intersection over unionbetween the predicted andground truth boxes.
# Applications of visual detection and recognition
Facial Recognition
People Counting
Self Driving Cars(once the image sensor detects any sign of a living being in its path,  it  automatically  stops)
Security(used by the government to access the security feed and match it with their existing database to find any criminalsor to detect the robbers’ vehicle)
# Yolo Algorithm
Yolo   is   an   algorithm   that   uses convolutionalneural   networks   for   visualdetection.So what's great about visualdetection? In comparison to recognition algorithms, a detection algorithm does not only predict class labels, but detects locations of visualsas well.YOLO  is  an  extremely  fast  real  time  multi  object  detection  algorithm.  YOLO  stands  for “You Only Look Once”. The algorithm applies a neural network to an entire image. The network divides the image into an S x S grid and comes up with bounding boxes, which are boxes drawn around images and predicted probabilities for each of these regions.The  method  used  to  come  up  with  these  probabilities  is  logistic  regression.  The  bounding boxes are weighted by the associated probabilities. For class prediction, independent logistic classifiers are used.
# Advantages and disadvantages of YOLO
YOLO  is  orders  of  magnitude  faster(45  frames  per  second)  than  other  object  detection algorithms.The limitation  of  YOLO  algorithm  is  that  it  struggles  with  small  objects  within  the  image, for example, it might have difficulties in detecting a flock of birds. This is due to the spatial constraints of the algorithm.
# Convolution Neural Network
Convolution Neural NetworkImage   &   Video   recognition,   Image   Analysis   &   Classification,   Media   Recreation, Recommendation   Systems,   Natural   Language   Processing,   etc.   The   advancements   in Computer  Vision  with  Deep  Learning  has  been  constructed  and  perfected  with  time, primarily over oneparticular algorithm — aConvolutional Neural Network.Convolutional  neural  networks  are  quite  different  from  most  other  networks.  They  are primarily used for image processing, but can also be used for other types of input, such as as audio.  A  typical use  case  for  CNNs  is  where  you  feed  the  network  images  and  it  classifies the data. CNNs tend to start with an input “scanner,” which is not intended to parse all of the training data at once. For example, to input an image of 100 x 100 pixels, you wouldn’twant a layer with 10,000 nodes. Rather, you create a scanning input layer of say, 10 x 10, and you feed the first 10 x 10 pixels of the image. Once you’ve passed that input, you feed it the next 10 x 10 pixels by moving the scanner one pixel to the right.This input data is  then fed through convolutional layers instead of  normal layers,  where  not all nodes  are  connected.  Each node  only concerns  itself  with close  neighboring cells.  These convolutional  layers  also  tend  to  shrink  as  they  become  deeper,  mostly  by  easily  divisible factors of the input. Beside these convolutional layers, they also often featurepooling layers.Pooling  is  a  way  to  filter  out  details:  a  commonly  found  pooling  technique  ismax  pooling,where we take, say, 2 x 2 pixels and pass on the pixel with the most amount of red.
# Tensorflow
Tensor   flowis  Google’s  Open  Source  Machine  Learning  Frameworkfor   dataflow programming across a range of tasks.Nodes in the graph represent mathematical operations, while  the  graph  edges  represent  the  multi-dimensional  data  arrays  (tensors)  communicated between them Tensors are just multidimensional arrays, an extension of 2-dimensional tables to data with a higher  dimension.  There  are  many  features  of  Tensor  flow  which  makes  it  appropriate  for Deep  Learning.  So,  without  wasting  any  time,  let’s  see  how  we  can  implement  Object Detection using Tensor flow.Tensor  Flow  provides  a  variety  of  different  tool  kits  that  allow  you  to  write  code  at  your preferred  level  of  abstraction.  For  instance,  you  can  write  code  in  the  Core  Tensor  Flow (C++) and call that method from Python code. You can also define the architecture on which your code should run (CPU, GPU etc.)Mostly Tensor Flow is used as a backend framework whose  modules  are  called  through  Keras  API.  Typically,  Tensor  Flow  is  used  to  solve complex problems like Image Classification, Object Recognition, Sound Recognition, etc
# Problems and Challenges
The major challenge in this problem is that of the variable dimension of the output which iscaused due to thevariable number of objects that can be present in any given inputimage.Any general machinelearningtask requires afixed dimension of input and outputforthemodel to be trained. Another important obstacle for widespread adoption of object detectionsystems is the requirement of real-time (>30fps) while being accurate in detection. The morecomplex the model is, themore time it requires for inference;and the less complex themodelis, the less is the accuracy. This trade-off between accuracy and performance needs tobe chosen as per the application. The problem involvesclassification as wellas regression,leading themodel to be learnt simultaneously. This addsto thecomplexity of the problem.
# Variable number of visuals
We already mentioned the part about a variable number of visuals, but we omitted why it’s a problem atall. When trainingmachine learningmodels, you usuallyneed torepresent data into fixed-sized vectors. Since the number of objects in the image isnot known beforehand, wewouldnot know thecorrect numberof outputs.Becauseof his, some post-processingis required, which adds complexity to the model.Historically,the variablenumber ofoutputshas been tackled using a sliding window based approach,  generating  the  fixed-sized   features of that window for all thedifferentpositions of it. After gettingall predictions, some are discarded and some are merged to get thefinal result.
# Sizing
Another big challengeis the different  conceivable   sizes   of   visuals. When doing simple classification,youexpect   and  want  to  classify  visuals that cover most of the image. On theother  hand,  some  of  the   visualsyou may want to find could be a small asadozen pixels ora small percentage   of  the   original  image. Traditionally thishas been solvedwith using sliding windows of different sizes, which is simple but very inefficient.
# Modeling
A third challenge is solving two problems at the same time. How do we combine the two differenttypes of requirements: location and classification into, ideally, a single model?
# Dependencies
To   build   Yolo   we're   going   to   need   Tensorflow   (deep   learning),   NumPy(numerical computation)  and  Pillow  (image  processing)  libraries.  Also  we're  going  to  use  seaborn's color  palette  for  bounding  boxes  colors.  Finally,  let's  import  IPython  functiondisplay()to display images in the notebook.<br/>
import tensorflow as tf<br/>
import numpy as np <br/>
from PIL import Image,ImageDraw,ImageFont<br/>
from IPython.display <br/>
import display from seaborn <br/>
import color_palette <br/>
import cv2.<br/>
# Model hyper parameters
Next, we define some configurations for Yolo.<br/>
_BATCH_NORM_DECAY = 0.9<br/>
_BATCH_NORM_EPSILON = 1e-05<br/>
_LEAKY_RELU = 0.1<br/>
_ANCHORS = [(10, 13), (16, 30), (33, 23),(30, 61), (62, 45), (59, 119),(116, 90), (156, 198), (373, 326)]<br/>
_MODEL_SIZE = (416, 416)_MODEL_SIZErefers to the input size of the model.<br/>
# Leaky ReLU
Leaky ReLU is a slight modification of ReLU activation function.
# REFERENCES
Towards  data  science  visual  detection  and  recognition.  An  introduction  to  implementing the YOLO algorithm for multi object detection in images<br/>
kaggle: visual detection and  recognition.import libraries and define functions for plotting the datausing Tenserflow, Convolutional neural network and Yolo.<br/>
