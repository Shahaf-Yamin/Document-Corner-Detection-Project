# Document-Corner-Detection-Project

This repository contains the implementation of my final project at the Introduction To Deep Learning Course, Electrical Engineering Departement, Ben Gurion University. 
Documents exist in both paper and digital form in our everyday life. Paper documents are easier to carry, read, and share whereas digital documents are easier to search, index, and store. For efficient knowledge management, it is often required to convert a paper document to digital format.   Document segmentation from the background is one of the major challenges of digitization from a natural image. An image of a document often contains the surface on which the document was placed to take the picture. It can also contain other non-document objects in the frame. Precise localization of document, often called document segmentation, from such an image is an essential first step required for digitization. Some of the challenges involved in localization are variations in document and background types, perspective distortions, shadows, and other objects in the image.

’ICDAR 2015 SmartDoc Challenge 1’ dataset is used for training and testing. The dataset contains 150 videos. Each video is around 10 seconds long and contains one of the six types of documents placed on one of the five distinct backgrounds. We split these videos into two disjoint sets.
Ground truth is available in the form of (x,y) coordinates labels:
top left (tl), top right (tr), bottom right (br), and bottom left (bl).

The short videos are taken over 5 different backgrounds, where background 5 is relatively complex as will be shown later.
Over each background, the videos have been taken over different document types - datasheet, letter, magazine, paper, patent, and tax, (5 short videos per background and document type).

# Final Results
## Background 01
![1](https://user-images.githubusercontent.com/49431639/156576701-881a5c99-8cee-4076-868a-b6c754bb1002.jpg)
## Background 02
![1](https://user-images.githubusercontent.com/49431639/156576860-c4d643a1-74b2-4911-ab91-83f48a0c83d8.jpg)
## Background 03
![1](https://user-images.githubusercontent.com/49431639/156576930-ef44c060-13c1-4f63-bf3b-2f32124bbc2c.jpg)
## Background 04
![1](https://user-images.githubusercontent.com/49431639/156576982-058884f5-e05c-4130-8b01-d803b67a6c36.jpg)
## Background 05
![1](https://user-images.githubusercontent.com/49431639/156576999-bf7bbb8c-23cc-4c0e-a59a-a4f748b6b9eb.jpg)



