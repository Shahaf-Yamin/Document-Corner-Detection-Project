# Document-Corner-Detection-Project

This repository contains the implementation of my final project at the Introduction To Deep Learning Course, Electrical Engineering Departement, Ben Gurion University. 
Documents exist in both paper and digital form in our everyday life. Paper documents are easier to carry, read, and share whereas digital documents are easier to search, index, and store. For efficient knowledge management, it is often required to convert a paper document to digital format.   Document segmentation from the background is one of the major challenges of digitization from a natural image. An image of a document often contains the surface on which the document was placed to take the picture. It can also contain other non-document objects in the frame. Precise localization of document, often called document segmentation, from such an image is an essential first step required for digitization. Some of the challenges involved in localization are variations in document and background types, perspective distortions, shadows, and other objects in the image.

Dataset
’ICDAR 2015 SmartDoc Challenge 1’ dataset is used for training and testing. The dataset contains 150 videos. Each video is around 10 seconds long and contains one of the six types of documents placed on one of the five distinct backgrounds. We split these videos into two disjoint sets. Ground truth is available in the form of (x,y) coordinates labels: top left (tl), top right (tr), bottom right (br), and bottom left (bl).
The short videos are taken over 5 different backgrounds, where background 5 is relatively complex as will be shown later. Over each background, the videos have been taken over different document types - datasheet, letter, magazine, paper, patent, and tax, (5 short videos per background and document type).
As an initial pre-process, we replaced each video with its frames (resolution of 1920X1080). The following images are random frames taken from different videos:


