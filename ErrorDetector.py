from IPython.display import display
import os
import random
import requests
from PIL import Image
import torch
#from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import numpy as np
import gc
import io
import base64
import gc
try:
    from . import Projection as prj
except:
    import projection as prj
import tempfile
import shutil
import copy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import csv



BodyRegionTextSkeleton=("The image I am sending is frontal projections of one CT scan, focusing on showing the skeleton. Look at it carefully, and answer the questions below:\n"
"Q1- Which bones are on the top of the image? Which bones are on its bottom?\n"
"Q2- Which of the following landmarks are present in the image? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"skull = _"
"neck = _"
"trachea = _"
"upper ribs = _"
"lower ribs = _"
"lumbar spine = _"
"pelvis = _"
"femurs = _\n"
"Q3- Considering these landmarks and the bones on the image top and bottom, give me a complete list of all organs (not bones) usually contained within this image limits (just list their names).\n"# In your list, carefully consider if the following organs are usually contained or not: liver, gallbladder, stomach, spleen, pancreas and kidneys. \n"
"Q4- Based on your answer to Q2 and Q3, is the %(organ)s usually present within this image limits? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"Q4 = _\n")


AorticArchTextSkeleton=("If you answered no to Q4, skip Q5 and Q6. If you answered yes, continue:\n"
                "Q5- What parts of the respiratory system are contained in the image region?\n"
                "Q6- Based on your answer to Q3, is the trachea usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
                "Q6 = _\n")


BodyRegionText=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It looks like an AP (anterior-to-posterior) X-ray. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and which organs and bones inside this region are crossing the image's upper boundary and its lower boundary. Present a complete list of all organs usually present in the body region contained in the image (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q2 = _\n")

AorticArchText=(
    "If you answered 'no' to Q2, skip Q3, Q4 and Q5. If you answered 'yes', continue.\n"
    "Q3: Based on your answer to Q1, are the lungs present in the image? \n"
    "Answer 'yes' or 'no' using the template below, substituting _ with Yes or No.\n"
    "Q3 = _\n"
    "If you answered 'no' to Q3, skip Q4 and Q5. If you answered 'yes', continue.\n"
    "Q4: Repeat from you answer to Q1, which organs and bones are crossing the image upper boundary (none, brain, neck, lungs, heart or others)? \n"
    "Q5: Considering your answer to Q3 and Q4, does the image include the aortic arch? The aortic arch is placed just above the heart. It is not present if you listed the heart, lower ribs, or lower lungs in Q4. It is present if you listed none, brain or neck. \n"#It is probably not present if you listed the lungs. \n"
    "Answer 'yes' or 'no' using the template below, substituting _ with Yes or No.\n"
    "Q5 = _\n"
)
#90% localization accuracy




DescendingAortaErrorDetectV0=('Consider the following questions, were positive answers indicate a correct annotation: '
    'Point 1: Consider the bones in the image. Is the lumbar spine visible? If it is, does the red shape come down to the lumbar spine height? '
    'Point 2: Is the red shape a single connected part? If you see two or more red shapes in the image, it is wrong.')

DescendingAortaErrorDetectV1="""Consider the following questions, were positive answers indicate a correct annotation: 
    Point 1: Does the red shape resemble a line? It does not matter if the line is curved or straight. Also, the line can have a round structure in the upper chest.
    Point 2: Is the red shape a single connected part? If you see two or more red shapes in the image, it is wrong."""
#70% acc in 2 shot correct annos

DescendingAortaErrorDetect=('Consider the following questions, were positive answers indicate a correct annotation: '
    'Point 1: Does the red shape resemble a line? It does not matter if the line is curved or straight. Also, the line may have a round bulge at the top, representing the aortic arch (upper chest).'
    'Point 2: Is the red shape a single connected part? If you see two or more red shapes in the image, it is wrong.')
FullAortaErrorDetect=DescendingAortaErrorDetect


FullAortaErrorDetectV0=('Consider the following questions, were positive answers indicate a correct annotation: '
                'Point 1: Does the red shape reach the thoracic region (high ribs), showing a curve in this area?'
                'Point 2: Consider the bones in the image. Is the lumbar spine visible? If it is, does the red shape come down to the lumbar spine height? '
                'Point 3: Is the red shape a single connected part? If you see two or more red shapes in the image, it is wrong. ')

PostcavaErrorDetect=('Consider the following questions, were positive answers indicate a correct annotation: '
            'Point 1: Does the red shape reach the thoracic region (high ribs)?'
            'Point 2: Consider the bones in the image. Is the lumbar spine visible? If it is, does the red shape come down to the lumbar spine height? '
            'Point 3: Is the red shape a single connected part? If you see two or more red shapes in the image, it is wrong. ')

KidneysErrorDetectOld=("Consider the following anatomical information: A person usually has two kidneys, check if the image display one, two or more red objects, this is a very important point. "
                      "Each kidney has a bean-shaped structure, with a slightly concave surface facing the spine, and a clearly convex surface facing outward. Check if the red objects resemble this shape and are complete. "
                      " The kidneys are located on either side of the spine, at the level of the lower ribs. Check if the red objects, if a pair, are on either side of the spine and at the level of the lower ribs. \n")

LiverErrorDetect=("When evaluating the overlays, consider the following anatomical information:\n"
"a) The liver is a large organ, with triangular or wedge-like shape.\n"
"b) The liver is located in the upper right quadrant of the abdomen (left of the figure, like an AP X-ray), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. The liver is not near the pelvis.\n"
"c) The liver position is primarily under the rib cage. The overlay must show red in the ribs region. \n")


KidneysErrorDetect=("When evaluating the overlays, consider the following anatomical information:\n"
"a) A person usually has two kidneys, check if the image display one, two or more red objects.\n"
"b) Each kidney has a bean-shaped structure, with a slightly concave surface facing the spine, and a clearly convex surface facing outward. Check if the red objects resemble this shape.\n"
"c) The kidneys are located on either side of the spine, at the level of the lower ribs. Check if the red objects are on either side of the spine and at the level of the lower ribs. \n")


StomachErrorDetect="""Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""

PancreasErrorDetectV0="""Consider the following anatomical information:
a) Shape: The pancreas is an elongated organ with a tadpole-like shape. The pancreas head is its thickest part and points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray. The other side of the pancreas is thin.
b) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. The organ is mostly horizontal, but may be slightly curved and its head usually sits lower than its tail.
C) Smoothness: The pancreas is a single smooth shape and it does not have very sharp edges."""

SpleenErrorDetect="""Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural curve of the spleen, with no small recesses.
b) Shape 2: The spleen red overlay should not have irregular or random shapes. It should not include internal gaps or sharp, angular points.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney."""

GallbladderErrorDetect="""Consider the following anatomical information:
a) Shape: The gallbladder red overlay should be pear-shaped or an elongated curved sack.
b) Shape 2: The gallbladder red overlay should be smooth. It should not have many random points.
c) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""

PancreasErrorDetect="""When evaluating the overlays, consider the following anatomical information:
a) Shape: Check if the red overlay resembles the overall shape of a pancreas.
b) Head: The pancreas is the thickest on its head, which points to the right side of the human body or the left side of the image, which is oriented like an AP X-ray.
c) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
d) Smoothness: The pancreas is a single smooth shape and it does not have very sharp edges."""

DescriptionsErrorDetect={
    "aorta":FullAortaErrorDetect,
    "descending aorta":DescendingAortaErrorDetect,
    "postcava":PostcavaErrorDetect,
    "liver":LiverErrorDetect,
    "kidneys":KidneysErrorDetect,
    "spleen":SpleenErrorDetect,
    "stomach":StomachErrorDetect,
    "pancreas":PancreasErrorDetect,
    "gall_bladder":GallbladderErrorDetect}

StomachErrorDetectShapeless="""Consider the following anatomical information, you must ignore the shape of the overlay, and evaluate only the location and unity of the red overlay:
a) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""

GallbladderErrorDetectShapeless="""Consider the following anatomical information, you must ignore the shape of the overlay, and evaluate only the location and unity of the red overlay:
a) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""

PancreasErrorDetectShapeless="""Consider the following anatomical information, you must ignore the shape of the overlay, and evaluate only the location and unity of the red overlay:
a) Unity: The pancreas red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Location: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage."""


DescriptionsErrorDetectShapeless={
    "aorta":FullAortaErrorDetect,
    "descending aorta":DescendingAortaErrorDetect,
    "postcava":PostcavaErrorDetect,
    "liver":LiverErrorDetect,
    "kidneys":KidneysErrorDetect,
    "spleen":SpleenErrorDetect,
    "stomach":StomachErrorDetectShapeless,
    "pancreas":PancreasErrorDetectShapeless,
    "gall_bladder":GallbladderErrorDetectShapeless}























ZeroShotInstructions=("The image I am sending is a frontal projection of a CT scan. "
                      "It is not a CT slice, we have transparency and can see through the entire body, "
                      "like a X-ray. The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the image should be marked in red, "
                      "using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. Check if the red region is coherent with "
                      "the expected shape and location of a %(organ)s.")

ZeroShotInstructionsShapeless=("The image I am sending is a frontal projection of a CT scan. "
                      "It is not a CT slice, we have transparency and can see through the entire body, "
                      "like a X-ray. The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the image should be marked in red, "
                      "using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s location.")

FindErrors=("The image I am sending now, we can call it Image %(number)s, is a frontal projection of a CT scan. "
                      "It is not a CT slice, we have transparency and can see through the entire body, "
                      "like a X-ray. The left side of the image represents the right side of the human body, it looks like an AP (anterior-to-posterior) X-ray. "
                      "The %(organ)s region in the image should be marked in red, "
                      "using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. Check if the red region is coherent with "
                      "the expected shape and location of a %(organ)s, and analyze potential mistakes, if any.")

FindErrorsShapeless=("The image I am sending now, we can call it Image %(number)s, is a frontal projection of a CT scan. "
                      "It is not a CT slice, we have transparency and can see through the entire body, "
                      "like a X-ray. The left side of the image represents the right side of the human body, it looks like an AP (anterior-to-posterior) X-ray. "
                      "The %(organ)s region in the image should be marked in red, "
                      "using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s location.")

FindErrorsSkeleton=("The image I am sending now, we can call it Image %(number)s, is a frontal projection of a CT scan. "
                      "It is not a CT slice, we have transparency and can see through the entire body, "
                      "like a X-ray. However, I have confugured the image to mostly display the bones. "
                      "The left side of the image represents the right side of the human body, it looks like an AP (anterior-to-posterior) X-ray. "
                      "The %(organ)s region in the image should be marked in red, "
                      "using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. Check if the red region is coherent with "
                      "the expected shape and location of a %(organ)s, and analyze potential mistakes, if any.")

SummarizeInstructions=("Summarize your last answer, using only 2 words: "
                       "'good annotation' or 'bad annotation'.")
SummarizeInstructionsFewShot=("Summarize your last answer, using only 2 words: "
                            "'good annotation' or 'bad annotation'. "
                            "Is the annotation for the last image I sent a 'good annotation' or a 'bad annotation'?")

CompareSummarizeED=("The text below represents an evaluation of an overlay. The overlay was a red shape over a CT scan projection, which was an image similar to an X-ray. "
                "The overlay was supposed to mark the %(organ)s region, being an annotation for the organ. "
                "A VLM like you analyzed the overlay and evaluated if it was a 'good annotation' or a 'bad annotation'. Its answer is the text below."
                "I want you ro read the VLM's answer and tell me if the VLM considered the annotation good or bad. Answer me with only these words: 'good annotation' or 'bad annotation'. "
                "The text is:\n")

OneShotFirstPart=("The images I am sending are a frontal projections of CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. They are oriented like AP X-rays, "
                      "i.e., the left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. You must analyze if the red region is coherent with "
                      "the expected shape and location of a %(organ)s."
                      "However, now I am sending you just an example image, which is a 'good annotation'. "
                      "I will send you the image for evaluation after you see the example. "
                      "Take a good look and learn with it.")

OneShotFirstPartShapeless=("The images I am sending are a frontal projections of CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. They are oriented like AP X-rays, "
                      "i.e., the left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s location."
                      "However, now I am sending you just an example image, which is a 'good annotation'. "
                      "I will send you the image for evaluation after you see the example. "
                      "Take a good look and learn with it.")

FewShotFirstPart=("The images I am sending are a frontal projections of CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. They are oriented like AP X-rays, "
                      "i.e., the left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. You must analyze if the red region is coherent with "
                      "the expected shape and location of a %(organ)s."
                      "However, now I am sending you just an example image, which is a 'good annotation'. "
                      "I will send mutiple examples in diverse prompts. "
                      "Take a good look and learn with the examples.")

FewShotFirstPartShapeless=("The images I am sending are a frontal projections of CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. They are oriented like AP X-rays, "
                      "i.e., the left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s location."
                      "However, now I am sending you just an example image, which is a 'good annotation'. "
                      "I will send mutiple examples in diverse prompts. "
                      "Take a good look and learn with the examples.")

OneShotSecondPart=("I am sending you now the image for evaluaiton. Check if it has a 'good annotation' or a 'bad annotation'. "
                  "The example image should help you evaluate it. Think throughly and provide a detailed explanation for "
                  "why the last image represents a 'good annotation' or a 'bad annotation'.")


FewShotSecondPart=("I am sending you now the image for evaluaiton. Check if the red overlay on it is a 'good annotation' or a 'bad annotation'. "
                  "The example images should help you evaluate it. Think throughly and provide a detailed explanation for "
                  "why the last image represents a 'good annotation' or a 'bad annotation'.")
                  

#OneShotInstructions=("The image I am sending are a frontal projections of a CT scans. "
#                     "It is not CT slices, instead, they have transparency and let you see throgh "
#                     "the entire human body, like an X-ray does. The left side of the image represents the right side of the human body. The %(organ)s region in the images should "
#                     "be marked in red, using an overlay. However, some images present correct overlays, "
#                     "while others present incorrect overlays. Image 1 is an example of a good %(organ)s annotation."
#                     "I want you to evaluate the correctness of the %(organ)s annotation in the second image. "
#                     "The example (Image 1) should help you evaluate it. Think throughly and provide a detailed explanation for "
#                     "why the last image represents a 'good annotation' or a 'bad annotation'.")

#FewShotInstructions=("The images I am sending are a frontal projections of a CT scans. "
#                     "They are not CT slices, instead, they have transparency and let you see throgh "
#                     "the entire human body, like an X-ray does. The left side of the image represents the right side of the human body. The %(organ)s region in the images should "
 #                    "be marked in red, using an overlay. However, some images present correct overlays, "
 #                    "while others present incorrect overlays. The first %(examples)s are examples, and "
 #                    "I will inform you (below) if each one of them is a 'good annotation' or a 'bad annotation': \n")

#FewShotInstructionsEnd=("I want you to evaluate the correctness of the %(organ)s annotation in the last image, %(last)s. "
#                        "The examples should help you evaluate it. Think throughly and provide a detailed explanation for "
#                        "why image %(last)s represents a 'good annotation' or a 'bad annotation'.")

liver=(" Consider the following anatomical information: the liver is a large, triangular organ located in the upper right quadrant of the abdomen, "
      "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
      " into the left upper quadrant. "
      "The liver position is primarily under the rib cage. In assessing the annotation, I want you to answer the following questions: \n"
      "1. Is the red overlay a single contiguous object? \n"
      "2. Does the shape of the red overlay resemble the typical triangular or wedge-like shape of the liver? \n"
      "3. Is the red overlay primarily located in the upper right quadrant of the abdomen, just below the diaphragm? \n")
      
kidneys_questions=("Consider the following anatomical information: the kidneys are two bean-shaped organs located on either side of the spine. "
         "Each kidney should appear as a distinct structure. "
         "The kidneys are located around the T12 to L3 vertebrae, primarily in the upper abdomen."
         "In assessing the annotation and finding potential errors on it, I want you to answer the following questions: \n"
         "1. Is the red overlay divided into two distinct regions, one for each kidney? \n"
         "2. Does the shape of the red overlay resemble the typical bean shape of each kidney? \n"
         "3. Is the red overlay located in the correct anatomical region, on either side of the spine and close to the posterior wall? \n")


liver_no_question=(" Consider the following anatomical information: the liver is a large, triangular organ located in the upper right quadrant of the abdomen, "
      "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
      " into the left upper quadrant. "
      "The liver position is primarily under the rib cage. \n")

liver_describe=(" Consider the following anatomical information: the liver is a large, triangular organ located in the upper right quadrant of the abdomen, "
      "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
      " into the left upper quadrant. "
      "The liver position is primarily under the rib cage. \n"
      "Throuhgly describe the overlay: what is its shape? where is it? Then you say if it corresponds to the liver or not. ")



KidneysDescriptionED=("Consider the following anatomical information: A person usually has two kidneys, check if the image display one, two or more red objects, this is a very important point. "
                      "Each kidney has a bean-shaped structure, with a slightly concave surface facing the spine, and a clearly convex surface facing outward. Check if the red objects resemble this shape and are complete. "
                      " The kidneys are located on either side of the spine, at the level of the lower ribs. Check if the red objects, if a pair, are on either side of the spine and at the level of the lower ribs. \n")


AdrenalGlandDescriptionED=("Consider the following anatomical information:\n"
                   "a) Number: a person usually has two adrenal glands, one on top of each kidney.\n"
                   "b) Location: the adrenal glands are located on the superior aspect of each kidney, in the retroperitoneal space.\n"
                   "c) Right Adrenal Gland Shape (right side of the body): triangular shape.\n"
                   "d) Left Adrenal Gland Shape (left side of the body): Generally crescent-shaped or semilunar. May appear as a curved line or elongated structure above the kidney.\n"
                   "e) Size: adrenal glands are relatively small compared to the kidneys.\n")

AortaDescriptionED = (
    "Consider the following anatomical information:\n"
    "a) Shape: The aorta should appear as a long vertical red line with a curve at the top, resembling a question mark (?). The aorta should either display this curve at the top, or extend as far as possible in the top of the image, in case the heart is not visible in the image. The aorta starts considerably higher than the diaphragm and the lung bases. Check if the the shape of the red overlay resembles this description.\n"
    "b) Location: After the small curve at the top (aortic arch), the aorta should run parallel to the spine, which usually appears as a vertical line along the midline of the body. Check if the red overlay has the correct position and shape of the aorta relative to the spine.\n"
    "c) Completeness: The aorta should be visible from the heart down to the pelvis (or down to the image bottom, if the pelvis is not visible). Check if the red overlay accurately shows the full length of the aorta from the heart to the pelvis."
)


DescendingAortaDescriptionED = (
 "Closely follow the following instructions in evauating the image:\n"
 "1- Do you see a red overlay in the image? \n"
 "2- Does the red overlay resemble a long, vertical line approximatelly in the center of the body? It may be slightly curved if the spine is curved. The aortic arch is not present. \n" #
 "3- Does the red line touch the top of the image? it MUST touch. This is very important, check carefully. \n"
 "4- Does the red line extend as far as the end of the lumbar spine (if it is visible in the image)? \n"
)
#50 %


LiverDescriptionED=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
"a) The liver is a large organ, with triangular or wedge-like shape.\n"
"b) The liver is located in the upper right quadrant of the abdomen (left of the figure, like an AP X-ray), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. The liver is not near the pelvis.\n"
"c) The liver position is primarily under the rib cage. The overlay must show red in the ribs region. \n")
#"d) The liver is a single structure.\n")


SpleenDescriptionED=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
                    "a) The spleen is an organ with an oval or bean-like shape, approximately the size of a fist.\n"
                    "b) The spleen is located in the upper LEFT quadrant of the abdomen. I.e., in the right side of the figure, which is oriented like an AP X-ray."
                    "c) The spleen is located beneath the diaphragm and behind the stomach. It lies at the level of the 9th to 11th ribs.\n"
                    "d) The spleen is situated near the LEFT kidney.")


StomachDescriptionED="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, or a sac with a downwards curvature.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay should be a single connected structure. If it has multiple structures or small disconnected parts, the overlay has a big error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
# 83% 3 rejections (10/12)
# Conservative: 100%, 10 rejections


PancreasDescriptionED="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The pancreas is an elongated organ with a tadpole-like shape. The pancreas head is its thickest part and points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray. The other side of the pancreas is thin.
b) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. The organ is mostly horizontal, but may be slightly curved and its head usually sits lower than its tail.
C) Smoothness: The pancreas is a single smooth shape and it does not have very sharp edges."""


GallbladderDescriptionED="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The gallbladder is a small, pear-shaped organ.
b) The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray).
c) The gallbladder sits near the lower edge of the liver and may overlap with the liver in this frontal CT projection.
d) The gallbladder should be represented as a single red object."""


LiverDescriptionLocation=("The liver is positioned at the height of the lower ribs, spanning from about the 7th to the 11th rib on the right side. In relation to the spine, the liver typically starts around the level of the 7th or 8th thoracic vertebra (T7–T8) and extends down to around the 1st or 2nd lumbar vertebra (L1–L2). In terms of the pelvis, the liver is well above it, separated by several vertebrae and the abdominal cavity. Its lower edge does not reach as far down as the pelvis, staying contained in the upper abdomen.")
GallbladderDescriptionLocation=("It is positioned beneath the liver, close to the lower edge of the rib cage. "
                                "It is typically found at the level of the 9th or 10th rib."
                                "The gallbladder is usually present when the lower ribs are present.")

DescriptionsED={
    "aorta":AortaDescriptionED,
    "descending aorta":DescendingAortaDescriptionED,
    "liver":LiverDescriptionED,
    "kidneys":KidneysDescriptionED,
    "adrenal_glands":AdrenalGlandDescriptionED,
    "spleen":SpleenDescriptionED,
    "stomach":StomachDescriptionED,
    "pancreas":PancreasDescriptionED,
    "gall_bladder":GallbladderDescriptionED}

StomachDescriptionEDShapeless="""When evaluating and comparing the overlays, consider only the overlay unity and location, do NOT consider its shape:
a) Unity: The stomach red overlay should be a single connected structure. If it has multiple structures or small disconnected parts, the overlay has a big error.
b) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""

PancreasDescriptionEDShapeless="""When evaluating and comparing the overlays, consider only the overlay unity and location, do NOT consider its shape:
a) Location: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. The organ is mostly horizontal, but may be slightly curved and its head usually sits lower than its tail.
b) Unity: The pancreas red overlay should be a single connected structure. If it has multiple structures or small disconnected parts, the overlay has a big error."""


GallbladderDescriptionEDShapeless="""When evaluating and comparing the overlays, consider only the overlay unity and location, do NOT consider its shape:
a) Unity: The gallbladder red overlay should be a single connected structure. If it has multiple structures or small disconnected parts, the overlay has a big error.
b) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""

DescriptionsEDShapeless={
    "aorta":AortaDescriptionED,
    "descending aorta":DescendingAortaDescriptionED,
    "liver":LiverDescriptionED,
    "kidneys":KidneysDescriptionED,
    "adrenal_glands":AdrenalGlandDescriptionED,
    "spleen":SpleenDescriptionED,
    "stomach":StomachDescriptionEDShapeless,
    "pancreas":PancreasDescriptionEDShapeless,
    "gall_bladder":GallbladderDescriptionEDShapeless}


organ_descriptions={'liver':liver}

def superpose_images(image1, image2):
    # Load images using PIL
    image1 = Image.open(image1)
    image2 = Image.open(image2)

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    # Check if images have 4 channels (RGBA)
    if image1_array.shape[-1] == 4:
        image1_array = image1_array[:, :, :3]  # Drop alpha channel if present
    if image2_array.shape[-1] == 4:
        image2_array = image2_array[:, :, :3]  # Drop alpha channel if present

    # Ensure the images have 3 channels (RGB)
    if image1_array.shape[-1] != 3 or image2_array.shape[-1] != 3:
        raise ValueError("Both images must be RGB with 3 channels.")

    # Get red, green, and blue channels from image 1 and image 2
    red1 = image1_array[:, :, 0]  # Red channel of image 1
    green1 = image1_array[:, :, 1]  # Green channel of image 1
    blue1 = image1_array[:, :, 2]   # Blue channel of image 1

    red2 = image2_array[:, :, 0]  # Red channel of image 2
    green2 = image2_array[:, :, 1]  # Green channel of image 2
    blue2 = image2_array[:, :, 2]   # Blue channel of image 2

    # Annotation masks for image 1 and image 2
    mask1 = (red1 != green1) & (red1 != blue1)
    mask2 = (red2 != green2) & (red2 != blue2)
    overlap = mask1 & mask2

    #create grey image
    image2_array[mask2,1] = image2_array[mask2,0]
    image2_array[mask2,2] = image2_array[mask2,0]
    grey=image2_array.copy()


    #annotation 1 red
    image2_array[mask1, 0] = grey[mask1, 0]  
    image2_array[mask1, 1] = 0  
    image2_array[mask1, 2] = 0 

    #annotation 2 yellow
    image2_array[mask2, 0] = grey[mask2, 0]  
    image2_array[mask2, 1] = grey[mask2, 1]  
    image2_array[mask2, 2] = 0 

    # Annotation overlap: red + yellow = orange
    image2_array[overlap, 0] = grey[overlap, 0]  
    image2_array[overlap, 1] = grey[overlap, 1] / 2   
    image2_array[overlap, 2] = 0    

    # Convert the modified array back to an image
    result_image = Image.fromarray(image2_array)

    return result_image

def SolidOverlay(image1):
    if isinstance(image1, Image.Image):
        image1_array = np.array(image1)
    else:
        image1_array=image1
    # Check if images have 4 channels (RGBA)
    if image1_array.shape[-1] == 4:
        image1_array = image1_array[:, :, :3]  # Drop alpha channel if present


    # Get red, green, and blue channels from image 1 and image 2
    red1 = image1_array[:, :, 0]  # Red channel of image 1
    green1 = image1_array[:, :, 1]  # Green channel of image 1
    blue1 = image1_array[:, :, 2]   # Blue channel of image 1

    # Annotation masks for image 1 and image 2
    mask1 = (red1 != green1) & (red1 != blue1)

    image1_array[mask1,0]=255
    image1_array[mask1,1]=0
    image1_array[mask1,2]=0

    # Convert the modified array back to an image
    result_image = Image.fromarray(image1_array)

    return result_image

def resize_image(img, size):
    # Get the original dimensions
    width, height = img.size
    
    # Determine the scaling factor based on the largest axis
    if width > height:
        new_width = size
        new_height = int((size / width) * height)
    else:
        new_height = size
        new_width = int((size / height) * width)
    
    # Resize the image with the new dimensions
    img = img.resize((new_width, new_height))
    
    return img

def ZeroShot(img,processor,model,text=ZeroShotInstructions,organ='liver',size=None):
    text=text%{'organ':organ}
    print(text)
    if isinstance(img, str):
        img = Image.open(img)
    if size is not None:
        img=resize_image(img,size)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text}
                ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 400, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text

def ZeroShot2Steps(img,processor,model,text=ZeroShotInstructions,
                    text2=SummarizeInstructions,
                    organ='liver',size=None):
    text=text%{'organ':organ}
    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    
    if size is not None:
        img=resize_image(img,size)

    generated_text=ZeroShot(img=img,text=text,processor=processor,model=model,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},  # Placeholder for the first image
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": model_text}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text2}
            # No image for the second user input
        ],
    },
]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5

def OneShot(img,good_example,processor,model,text1=OneShotFirstPart,
            text2=OneShotSecondPart,organ='liver',size=None):
    text1=text1%{'organ':organ}

    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    if isinstance(good_example, str):
        good_example = Image.open(good_example)
    if size is not None:
        img=resize_image(img,size)
        good_example=resize_image(good_example,size)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text1},
                {"type": "image"}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": 'Thank you for the example. The liver looks correct.'},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text2},
                {"type": "image"}
                ],
        },
    ]
    print(conversation)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[good_example,img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 400, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text

def OneShot2Steps(img,good_example,processor,model,text1=OneShotFirstPart,
            text2=OneShotSecondPart,organ='liver',size=None):
    text1=text1%{'organ':organ}
    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    if isinstance(good_example, str):
        good_example = Image.open(good_example)
    if size is not None:
        img=resize_image(img,size)
        good_example=resize_image(good_example,size)
    
    generated_text=OneShot(img=img,good_example=good_example,text1=text1,processor=processor,
                           model=model,size=size,text2=text2)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text1},
            {"type": "image"}
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": 'Thank you for the example. The liver looks correct.'},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text2},
            {"type": "image"}
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": model_text}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": SummarizeInstructions}
            # No image for the second user input
        ],
    },
]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[good_example,img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5


def FewShot(img,good_examples,bad_examples,processor,model,
            text1=FewShotFirstPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
            text2=FewShotSecondPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
            organ='liver',organ_descriptions=organ_descriptions,
            size=None):
    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    if size is not None:
        img=resize_image(img,size)

    tmp=[]
    for good_example in good_examples:
        if isinstance(good_example, str):
            good_example = Image.open(good_example)
        if size is not None:
            good_example=resize_image(good_example,size)
        tmp.append(good_example)
    good_examples=tmp
    tmp=[]
    for bad_example in bad_examples:
        if isinstance(bad_example, str):
            bad_example = Image.open(bad_example)
        if size is not None:
            bad_example=resize_image(bad_example,size)
        tmp.append(bad_example)
    bad_examples=tmp
    del tmp

    text1=text1%{'organ':organ}

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text1},
                {"type": "image"}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Thank you for the example. The red overlay is indeed a 'good annotation' for {organ}."%{'organ':organ}},
                ],
        }
    ]

    last='good'
    examples=[]
    examples.append(good_examples[0])
    good_i=1
    bad_i=0

    for i in range(len(good_examples)+len(bad_examples)-1):
        if last=='good' and bad_i<len(bad_examples):
            current='bad'
        elif last=='bad' and good_i<len(good_examples):
            current='good'
        elif last=='good' and bad_i>=len(bad_examples):
            current='good'
        elif last=='bad' and good_i>=len(good_examples):
            current='bad'

        if current=='good':
            examples.append(good_examples[good_i])
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image is a 'good annotation' for {organ}."%{'organ':organ}},
                    {"type": "image"}
                    ],
            })
            conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thank you for the example. The red overlay is indeed a 'good annotation' for {organ}."%{'organ':organ}},
                    ],
            })
            good_i+=1
            last='good'
        else:
            examples.append(bad_examples[bad_i])
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image is a 'bad annotation' for {organ}."%{'organ':organ}},
                    {"type": "image"}
                    ],
            })
            conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thank you for the example. The red overlay is indeed a 'bad annotation' for {organ}."%{'organ':organ}},
                    ],
            })
            bad_i+=1
            last='bad'

    conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text2},
                {"type": "image"}
                ],
        })

    print(conversation)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=examples+[img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 200, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text,conversation,examples+[img]

def FewShot2Steps(img,good_examples,bad_examples,processor,model,
                  text1=FewShotFirstPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
                    text2=FewShotSecondPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
                    organ='liver',
                    summarize_instructions=SummarizeInstructionsFewShot,
                    prt=False,size=None):

    generated_text,conversation,images=FewShot(img=img,good_examples=good_examples,
                                               bad_examples=bad_examples,text1=text1,text2=text2,
                                               organ=organ,processor=processor,model=model,size=size)

    if prt:
        for image in images:
            display(image)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('First answer:',model_text)

    conversation.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": model_text}
                            ],
                        })
    
    conversation.append({   "role": "user",
                            "content": [
                                {"type": "text", "text": summarize_instructions}
                            ],
                        })
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5
    
def get_random_file_paths(folder_path, n, exclude_path,contains='overlay_axis_1'):
    # Get all file paths in the specified folder
    if isinstance(folder_path, list):
        all_files = [file for file in folder_path 
                    if (os.path.isfile(file) and (exclude_path not in file))]
    else:
        all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                    if (os.path.isfile(os.path.join(folder_path, file)) \
                    and (exclude_path not in file))]

    if isinstance(contains, str):
        all_files = [file for file in all_files if contains in file]
    elif isinstance(contains, list):
        for contain in contains:
            all_files = [file for file in all_files if contain in file]

    # Get n random file paths from the list
    random_files = random.sample(all_files, min(n, len(all_files)))

    print('Example files:',random_files)

    return random_files

#say people have annatomical variations and the liver must not look perfect? it you have too many FP only

def calculate_accuracy(answers_good, answers_bad):
    """
    Calculates the accuracy of binary predictions given two vectors:
    answers_good (should ideally be all 1s) and answers_bad (should ideally be all 0s).

    Parameters:
        answers_good (np.ndarray): A vector of binary predictions where the expected values are 1.
        answers_bad (np.ndarray): A vector of binary predictions where the expected values are 0.

    Returns:
        float: The accuracy of the predictions.
    """
    answers_good, answers_bad = np.array(answers_good), np.array(answers_bad)

    # Combine predictions
    predictions = np.concatenate((answers_good, answers_bad))

    # Create ground truth vector
    ground_truth = np.concatenate((np.ones_like(answers_good), np.zeros_like(answers_bad)))

    # Calculate the number of correct predictions
    correct_predictions = np.sum(predictions == ground_truth)

    # Calculate accuracy
    accuracy = correct_predictions / len(predictions)
    
    return accuracy

def ZeroShotSystematicEval(good_annos,bad_annos,model,processor,
                           text=ZeroShotInstructions):
    answers_good=[]
    for target in good_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_good.append(ZeroShot2Steps(img=target,
                        model=model,processor=processor,
                        text=text))
        print('Traget:',target,'Answer:',answers_good[-1])

    
    answers_bad=[]
    for target in bad_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_bad.append(ZeroShot2Steps(img=target,
                        model=model,processor=processor,
                        text=text))
        print('Traget:',target,'Answer:',answers_bad[-1])

    print('Correct good annotation: ',np.array(answers_good).sum(),'/',len(answers_good))
    print('Correct bad annotation: ',(1-np.array(answers_bad)).sum(),'/',len(answers_bad))

    print('Accuracy: ',calculate_accuracy(answers_good,answers_bad))

def OneShotSystematicEval(good_annos,bad_annos,model,processor,text1=OneShotFirstPart,
                          text2=OneShotSecondPart):
    answers_good=[]
    for target in good_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_good.append(OneShot2Steps(img=target,
                        good_example=get_random_file_paths(good_annos,
                                n=1,exclude_path=target)[0],
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_good[-1])

    
    answers_bad=[]
    for target in bad_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_bad.append(OneShot2Steps(img=target,
                        good_example=get_random_file_paths(good_annos,
                                                           n=1,exclude_path=target)[0],
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_bad[-1])

    print('Correct good annotation: ',np.array(answers_good).sum(),'/',len(answers_good))
    print('Correct bad annotation: ',(1-np.array(answers_bad)).sum(),'/',len(answers_bad))

    print('Accuracy: ',calculate_accuracy(answers_good,answers_bad))

def FewShotSystematicEval(good_annos,bad_annos,model,processor,
                          text1=FewShotFirstPart,text2=FewShotSecondPart,n=1):
    answers_good=[]
    for target in good_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_good.append(FewShot2Steps(img=target,
                        good_examples=get_random_file_paths(good_annos,
                                n=n,exclude_path=target),
                        bad_examples=get_random_file_paths(bad_annos,
                                n=n,exclude_path=target),
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_good[-1])

    
    answers_bad=[]
    for target in bad_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_bad.append(FewShot2Steps(img=target,
                        good_examples=get_random_file_paths(good_annos,
                                n=n,exclude_path=target),
                        bad_examples=get_random_file_paths(bad_annos,
                                n=n,exclude_path=target),
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_bad[-1])

    print('Correct good annotation: ',np.array(answers_good).sum(),'/',len(answers_good))
    print('Correct bad annotation: ',(1-np.array(answers_bad)).sum(),'/',len(answers_bad))

    print('Accuracy: ',calculate_accuracy(answers_good,answers_bad))




def LoadLLaVAOneVision7B(bits=4):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    if bits==8:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            load_in_4bit=True,
            device_map='auto',
        )
    elif bits==4:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            load_in_4bit=True,
            device_map='auto',
        )
    else:
        raise ValueError('Invalid bits value. Use 4 or 8.')
    processor = AutoProcessor.from_pretrained(model_id)
    return model,processor

def LoadLLaVAOneVision72B(bits=8):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
    if bits==8:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                    load_in_8bit=True,
                    device_map='auto',
                    #use_flash_attention_2=True
                )
    elif bits==4:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                    load_in_4bit=True,
                    device_map='auto',
                    #use_flash_attention_2=True
                )
    else:
        raise ValueError('Invalid bits value. Use 4 or 8.')
    processor = AutoProcessor.from_pretrained(model_id)
    return model,processor




def red_on_top(image_path, percentage=0.05):
    import numpy as np
    from PIL import Image

    # 1- Load the image
    pil_img=Image.open(image_path).convert('RGB')
    img = np.array(pil_img)
    
    # Create a red mask where red channel is greater than both green and blue
    red_mask = (img[:, :, 0] > img[:, :, 1]) & (img[:, :, 0] > img[:, :, 2])

    # Determine the number of rows corresponding to the top percentage of the image height
    top_rows = int(percentage * img.shape[0])

    # Get the red mask for the top portion of the image
    red_mask_top = red_mask[:top_rows, :]
    
    # Count the number of red pixels in the top portion
    red_top = np.sum(red_mask_top)
    
    if red_top > 10:
        return True
    else:
        plt.imshow(img)
        plt.show()
        return False


def ErrorDetectionLMDeployZeroShot(clean, y, 
                           base_url='http://0.0.0.0:8000/v1', size=512,
                           text_region=BodyRegionTextSkeleton, 
                           organ_descriptions=DescriptionsED,
                           instructions=ZeroShotInstructions,
                           text_summarize=CompareSummarizeED, organ='liver',
                           save_memory=True, location_window='skeleton',solid_overlay=False,
                           simple_prompt_ablation=False):
    """
    clean: X-ray withouth annotation
    y: X-ray with annotation
    """

    if not simple_prompt_ablation:
    
        #check if organ should be in image
        organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

        if organ=='aorta':
            if location_window=='skeleton':
                organRegion+=AorticArchTextSkeleton
            else:
                organRegion+=AorticArchText

        if organ=='liver':
            organRegion+=LiverDescriptionLocation
        if organ=='gall_bladder':
            organRegion+=GallbladderDescriptionLocation

        conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                    base_url=base_url, size=size)
        q='q2'
        if 'skeleton' in location_window:
            q='q4'

        AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])

        if not AnswerNo:
            if organ=='aorta':
                if 'skeleton' not in location_window:
                    if ('no' in answer.lower()[answer.lower().rfind('q6'):]):#no lungs
                        organ='descending aorta'
                        if not red_on_top(y):
                            print('Aorta does not have red on top')
                            return 0.0
                else:
                    if ('yes' in answer.lower()[answer.lower().rfind('q5'):]):#aortic arch present
                        organ='aorta'
                    else:
                        organ='descending aorta'
                        if not red_on_top(y):
                            print('Aorta does not have red on top')
                            return 0.0
        
        if AnswerNo:
            a=RedArea(y)
            print('Annotation should be zero, but it is not')
            if a==0:
                return 1.0
            else:
                return 0.0
        #else:
        #    if RedArea(y)==0:
        #        print('Annotation is empty, but it should not be.')
        #        return 0.0
        
    if save_memory or simple_prompt_ablation:
        conversation=[]

    #organ should be present in image. Let's evaluate the label y
    #Analyze image
    if not simple_prompt_ablation:
        text=instructions % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}+organ_descriptions[organ]
    else:
        text='The image I am sending you is a frontal projection of a CT scan, which looks like and is oriented like an AP X-ray. It has a red overlay representing the ' +organ.replace('_',' ').replace('gall bladder','gallbladder')+'. Considering the organ shape and position, evaluate if the red overlay is a good or a bad annotation for the '+organ.replace('_',' ').replace('gall bladder','gallbladder')+'. Justify your answer.'
    imgs=[y]
    conversation, answer = SendMessageLmdeploy(imgs,text=text, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay)
    
    #summarize answer
    text=text_summarize % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}+answer
    conversation, answer = SendMessageLmdeploy([], text=text, conver=[], base_url=base_url, size=size)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5
        
    
        
def ErrorDetectionLMDeployFewShot(clean, y, good_examples,bad_examples,
                           base_url='http://0.0.0.0:8000/v1', size=512,
                           text_region=BodyRegionTextSkeleton, 
                           organ_descriptions=DescriptionsED,
                           text_summarize=CompareSummarizeED, organ='liver',
                           save_memory=True, location_window='skeleton',solid_overlay=False,
                           print_examples=True,shapeless=False):
    """
    clean: X-ray withouth annotation
    y: X-ray with annotation
    """

    print('Good examples recevied:',good_examples)
    print('Bad examples recevied:',bad_examples)
    
    #check if organ should be in image
    organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if organ=='aorta':
        if location_window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    if organ=='liver':
        organRegion+=LiverDescriptionLocation
    if organ=='gall_bladder':
        organRegion+=GallbladderDescriptionLocation

    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    if 'skeleton' in location_window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' not in location_window:
            if ('no' in answer.lower()[answer.lower().rfind('q3'):answer.lower().rfind('q3')+7]):#no lungs
                organ='descending aorta'
                if not red_on_top(y):
                    print('Aorta does not have red on top')
                    return 0.0
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q6'):]):#aortic arch present
                organ='aorta'
                text_compare=Compare2ImagesFullAorta
                print('Full aorta')
            else:
                organ='descending aorta'
                if not red_on_top(y):
                    print('Aorta does not have red on top')
                    return 0.0
    
    if AnswerNo:
        a=RedArea(y)
        print('Annotation should be zero')
        if a==0:
            return 1.0
        else:
            return 0.0
    #else:
    #    if RedArea(y)==0:
    #        print('Annotation is empty, but it should not be.')
    #        return 0.0
        

    
    if save_memory:
        conversation=[]

    #organ should be present in image. Let's evaluate the label y
    #create few shot prompt
    if isinstance(organ_descriptions, dict):
        organ_description=organ_descriptions[organ]
    

    if len(good_examples)==1 and len(bad_examples)==0:
        final_instructions=OneShotSecondPart % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}+organ_description
        if shapeless:
            text=OneShotFirstPartShapeless % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
        else:
            text=OneShotFirstPart % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
        text=text+organ_description
        conversation=CreateConversation(img_file_list=good_examples, text=text, conver=conversation,size=size,prt=True,solid_overlay=solid_overlay,
                                  role='user')
        model_answer='Indeed, the red overlay in this example shows a good annotation for the '+organ+'.'
        conversation=CreateConversation(img_file_list=[], text=model_answer, conver=conversation,size=size,prt=True,solid_overlay=solid_overlay,
                                  role='assistant')
    else:
        final_instructions=FewShotSecondPart % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}+organ_description
        if shapeless:
            text=FewShotFirstPartShapeless % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
        else:
            text=FewShotFirstPart % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
        text=text+organ_description
        for i in range(len(good_examples)):
            if i>0:
                text='This image is another good annotation for the '+organ+'.'+organ_description
            conversation=CreateConversation(img_file_list=[good_examples[i]], text=text, conver=conversation,size=size,prt=print_examples,
                                            solid_overlay=solid_overlay,
                                            role='user')
            model_answer='Indeed, the red overlay in this example shows a good annotation for the '+organ+'.'
            conversation=CreateConversation(img_file_list=[], text=model_answer, conver=conversation,size=size,prt=print_examples,
                                            solid_overlay=solid_overlay,
                                            role='assistant')
            if i==0:
                text='This image exmlifies a bad annotation for the '+organ+'.'+organ_description
            if i>0:
                text='This image is another bad annotation for the '+organ+'.'+organ_description
            conversation=CreateConversation(img_file_list=[bad_examples[i]], text=text, conver=conversation,size=size,prt=print_examples,
                                            solid_overlay=solid_overlay,
                                            role='user')
            model_answer='Indeed, the red overlay in this example shows a bad annotation for the '+organ+'.'
            conversation=CreateConversation(img_file_list=[], text=model_answer, conver=conversation,size=size,prt=print_examples,
                                            solid_overlay=solid_overlay,
                                            role='assistant')

    #Analyze image
    imgs=[y]
    conversation, answer = SendMessageLmdeploy(imgs,text=final_instructions, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay)
    
    #summarize answer
    text=text_summarize % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}+answer
    conversation, answer = SendMessageLmdeploy([], text=text, conver=[], base_url=base_url, size=size)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5



def get_files(pth,file_list,anno_window,organ,window,best,file_structure='dual'):
    """
    pth: path to images
    file_structure: dual (good and bad annotations in one folder, y2 assumed to be the good one) or all_good (only takes y2 samples) or dual_bad (assumes y1 and y2 to be bad)
    """

    if file_list is not None:
        with open(file_list, 'r') as file:
            file_list = file.readlines()
        # Removing the newline characters at the end of each line (optional)
        file_list = [line.strip() for line in file_list]

    if os.path.isdir(os.path.join(pth,os.listdir(pth)[0])):
        #not composite folder
        files=[]
        for pid in os.listdir(pth):
            if not os.path.isdir(os.path.join(pth,pid)):
                continue
            clean=os.path.join(pth,pid,pid+'_ct_window_'+window+'_axis_1.png')
            overlay_base='overlay_window_'+anno_window+'_axis_1'
            overlay=[x for x in os.listdir(os.path.join(pth,pid)) if overlay_base in x]
            if len(overlay)>1:
                raise ValueError('Multiple overlays found:',overlay)
            if len(overlay)==0:
                continue
            #print('overlay base:',overlay_base)
            #print('overlay:',overlay)
            #print('pid:',pid)
            overlay=overlay[0]
            files.append([clean,os.path.join(pth,pid,overlay)])

        if file_structure=='all_good' or 'pick_good_only':
            return files,[]
        elif file_structure=='all_bad' or 'pick_bad_only':
            return [],files
        else:
            raise ValueError('Invalid file structure for non composite folders.')
        
    #check if folder has y1 and y2 files
    y1_present=False
    y2_present=False
    for target in os.listdir(pth):
        if 'y1.' in target:
            y1_present=True
        if 'y2.' in target:
            y2_present=True
        if y1_present and y2_present:
            break

    if (not y1_present) and (not y2_present):
        if file_structure not in ['all_good','all_bad','pick_good_only','pick_bad_only']:
            raise ValueError('No y1 or y2 files found in folder. Use all_good, all_bad, pick_good_only or pick_bad_only file_structure.')
        files=[]
        for target in os.listdir(pth):
            if file_list is not None:
                if target[:14] not in file_list:
                    continue
            
            if 'ct_window_bone_axis_1' not in target:
                continue
            clean=os.path.join(pth,target)
            bdmap=clean[clean.rfind('/')+1:clean.rfind('_ct_window_bone')]

            flag=False
            for file in os.listdir(pth):
                if bdmap in file and ('overlay_window_'+anno_window) in file:
                    y1=file
                    flag=True
                    y1=os.path.join(pth,y1)
                    break
            if not flag:
                print('bdmap:',bdmap)
                raise ValueError('No y1 file found for:',clean)

            if window=='skeleton':
                clean=clean.replace('ct_window_bone','ct_window_skeleton')

            files.append([clean,y1])
        print(file_structure)
        if file_structure=='all_good' or file_structure=='pick_good_only':
            return files,[]
        elif file_structure=='all_bad' or file_structure=='pick_bad_only':
            return [],files
        
    


    files_bad=[]
    files_good=[]
    if file_structure=='dual' or file_structure=='pick_bad_only' or file_structure=='pick_good_only':
        for target in os.listdir(pth):
            if file_list is not None:
                #print ('Target:',target[:14] )
                if target[:14] not in file_list:
                    #print('Skipping:',target)
                    continue
            
            if 'ct_window_bone_axis_1' not in target:
                continue
            clean=os.path.join(pth,target)

            y1=clean.replace('ct_window_bone','overlay_window_'+anno_window).replace('.png','_y1.png')
            y2=clean.replace('ct_window_bone','overlay_window_'+anno_window).replace('.png','_y2.png')

            if organ not in clean[clean.rfind('ct'):]:
                y1=y1.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')
                y2=y2.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')

            if window=='skeleton':
                clean=clean.replace('ct_window_bone','ct_window_skeleton')

            if best==2:
                good_file=y2
                bad_file=y1
            else:
                good_file=y1
                bad_file=y2

            if file_structure=='dual':
                files_bad.append([clean,bad_file])
                files_good.append([clean,good_file])
            elif file_structure=='pick_bad_only':
                files_bad.append([clean,bad_file])
            elif file_structure=='pick_good_only':
                files_good.append([clean,good_file])
    elif file_structure=='dual_bad':
        for target in os.listdir(pth):
            if file_list is not None:
                #print ('Target:',target[:14] )
                if target[:14] not in file_list:
                    #print('Skipping:',target)
                    continue
            
            if 'ct_window_bone_axis_1' not in target:
                continue
            clean=os.path.join(pth,target)

            y1=clean.replace('ct_window_bone','overlay_window_'+anno_window).replace('.png','_y1.png')
            y2=clean.replace('ct_window_bone','overlay_window_'+anno_window).replace('.png','_y2.png')

            if organ not in clean[clean.rfind('ct'):]:
                y1=y1.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')
                y2=y2.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')

            if window=='skeleton':
                clean=clean.replace('ct_window_bone','ct_window_skeleton')
                #print('clean:',clean)

            files_bad.append([clean,y1])
            files_bad.append([clean,y2])
    elif file_structure=='all_good':
        for target in os.listdir(pth):
            if file_list is not None:
                #print ('Target:',target[:14] )
                if target[:14] not in file_list:
                    #print('Skipping:',target)
                    continue
            
            if 'ct_window_bone_axis_1' not in target:
                continue
            clean=os.path.join(pth,target)

            y1=clean.replace('ct_window_bone','overlay_window_'+anno_window).replace('.png','_y1.png')
            y2=clean.replace('ct_window_bone','overlay_window_'+anno_window).replace('.png','_y2.png')

            if organ not in clean[clean.rfind('ct'):]:
                y1=y1.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')
                y2=y2.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')

            if window=='skeleton':
                clean=clean.replace('ct_window_bone','ct_window_skeleton')
                #print('clean:',clean)

            files_bad.append([clean,bad_file])
    
    else:
        raise ValueError('Invalid file_structure value. Use dual or all_good or dual_bad or pick_good_only or pick_bad_only.')

    #print('Files good:',files_good)
    #print('Files bad:',files_bad)

    return files_good,files_bad


def ZeroShotErrorDetectionSystematicEvalLMDeploy(pth,
                                   base_url='http://0.0.0.0:8000/v1', size=512,
                                    text_region=BodyRegionTextSkeleton, 
                                    organ_descriptions=DescriptionsErrorDetect,
                                    instructions=ZeroShotInstructions,
                                    text_summarize=CompareSummarizeED, organ='liver',
                                    save_memory=True, location_window='skeleton',solid_overlay=False,
                                    anno_window='bone',best=2,
                                    file_list=None,
                                    file_structure='dual',
                                    dice_threshold=0.5,dice_threshold_max=0.9,
                                    dice_check=False,
                                    limit=None,skip_good=False,skip_bad=False,
                                    csv_file=None,restart=False,dice_list=None,
                                    shapeless=False,
                                    simple_prompt_ablation=False,min_cases=100,max_cases=10000):
        
        if shapeless:
            organ_descriptions=DescriptionsErrorDetectShapeless
            instructions=ZeroShotInstructionsShapeless
        
        if solid_overlay=='auto':
            if organ in ['aorta','postcava']:
                solid_overlay=True
            else:
                solid_overlay=False


        if csv_file is not None:
            column_names = ['case', 'answer', 'label', 'correct', 'organ']
            if os.path.exists(csv_file) and restart:
                os.remove(csv_file)
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)
            elif not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)

        #if dice_list is not None:
        #    import pandas as pd
        #    print('Loading dice list:',dice_list)
        #    df=pd.read_csv(dice_list,header=None, names=["case", "dice"])#

            #change dice threshold according to mean and std of dice in list
        #    a=df['dice'].mean()-df['dice'].std()
        #    #order dice in ascending order and get the 100th value
        #    b=df['dice'].sort_values().iloc[100]
        #    dice_threshold=max(dice_threshold,min(a,b))
        #    dice_threshold=min(dice_threshold,dice_threshold_max)
        #    print('Dice threshold re-defined to:',dice_threshold)

        #    filtered_df = df[df["dice"] < dice_threshold]
        #    # Get the list of 'case' values where 'dice' is below the threshold
        #    cases_below_threshold = filtered_df["case"].tolist()
        #    print('Number of cases below threshold:',len(cases_below_threshold))
        #    print('Cases below threshold:',cases_below_threshold)
            
        if dice_list is not None:
            import pandas as pd
            print('Loading dice list:',dice_list)
            df=pd.read_csv(dice_list,header=None, names=["case", "dice"])

            #change dice threshold according to mean and std of dice in list
            #a=df['dice'].mean()-df['dice'].std()
            #order dice in ascending order and get the 100th value
            #b=df['dice'].sort_values().iloc[100]
            #dice_th=max(dice_th,min(a,b))
            dice_th=min(df['dice'].mean()-df['dice'].std(),dice_threshold_max)
            print('Dice threshold re-defined to:',dice_th)

            filtered_df = df[df["dice"] < dice_th]
            # Get the list of 'case' values where 'dice' is below the threshold
            cases_below_threshold = filtered_df["case"].tolist()
            cases_high_dice=df[df["dice"] >= 0.95]["case"].tolist()
            if len(cases_below_threshold)<min_cases:
                #if too few cases, get 100 lowest dice cases to compare
                dice_th=df['dice'].sort_values().iloc[min_cases]
                #dice_th=min(dice_th,dice_threshold_max)
                #filtered_df = df[df["dice"] < dice_th]
                # Get the list of 'case' values where 'dice' is below the threshold
                cases_below_threshold = filtered_df["case"].tolist()
            if len( cases_below_threshold)>max_cases:
                #limit to 10% of dataset
                dice_th=df['dice'].sort_values().iloc[500]
                filtered_df = df[df["dice"] < dice_th]
                # Get the list of 'case' values where 'dice' is below the threshold
                cases_below_threshold = filtered_df["case"].tolist()

            print('Number of cases to check:',len(cases_below_threshold))

        answers=[]
        labels=[]
        outputs={}

        print('Getting files')
        files_good,files_bad=get_files(pth,file_list,anno_window,organ,location_window,best,file_structure=file_structure)
        print('Number of good files:',len(files_good))
        print('Number of bad files:',len(files_bad))    
        #sort files
        files_good.sort()
        files_bad.sort()

        #print('Good files:',files_good)
        #print('Bad files:',files_bad)
        #print('Files:',files_good+files_bad)
        #print('path:',pth)
        #print('files in path:',os.listdir(pth))

        for i in list(range(max(len(files_good),len(files_bad)))):
            skip_good=False
            skip_bad=False
            if dice_check:
                if dice_list is None:
                    good_file=files_good[i][1]
                    bad_file=files_bad[i][1]
                    dice=check_dice(good_file,bad_file)
                    high_dice=dice>dice_threshold
                    #print('Dice:',dice)
                else:
                    clean=files_good[i][0]
                    #print(clean[clean.rfind('/')+1:].replace('ct_window_skeleton','ct_window_bone'))
                    #print(clean.replace('ct_window_skeleton','ct_window_bone'))
                    #raise ValueError('Check dice')

                    if '/' in clean:
                        high_dice=clean[clean.rfind('/')+1:].replace('ct_window_skeleton','ct_window_bone') not in cases_below_threshold
                    else:
                        high_dice=clean.replace('ct_window_skeleton','ct_window_bone') not in cases_below_threshold

                    #print('clean:',clean)
                    #print('cases_below_threshold:',cases_below_threshold)
                
                if high_dice:
                    continue
            if i<len(files_good):
                clean,good_file=files_good[i]
                if ((csv_file is not None) and (not restart)):
                    if check_case_exists(csv_file, good_file):
                        print('Already exists, skipping case ',good_file)
                        skip_good=True
                if not skip_good:
                    print('Case below:',good_file)
                    answer_good=ErrorDetectionLMDeployZeroShot(clean=clean, y=good_file, 
                                base_url=base_url, size=size,
                                text_region=text_region, 
                                organ_descriptions=organ_descriptions,
                                instructions=instructions,
                                text_summarize=text_summarize, organ=organ,
                                save_memory=save_memory, location_window=location_window,
                                solid_overlay=solid_overlay,simple_prompt_ablation=simple_prompt_ablation)
                    
                    answers.append(answer_good)
                    labels.append(1.0)
                    outputs[good_file]=answer_good

                    print('Answer:',answer_good,'Label:',1.0,'Correct:',answer_good==1.0)
                    if csv_file is not None:
                        row = [good_file, answer_good, 1.0, answer_good==1.0, organ]
                        # Append the row to the CSV file
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

                    del answer_good
            
            if i<len(files_bad):
                clean,bad_file=files_bad[i]
                if ((csv_file is not None) and (not restart)):
                    if check_case_exists(csv_file, bad_file):
                        print('Already exists, skipping case ',good_file)
                        skip_bad=True
                if not skip_bad:
                    print('Case below:',bad_file)
                    answer_bad=ErrorDetectionLMDeployZeroShot(clean=clean, y=bad_file, 
                                base_url=base_url, size=size,
                                text_region=text_region, 
                                organ_descriptions=organ_descriptions,
                                instructions=instructions,
                                text_summarize=text_summarize, organ=organ,
                                save_memory=save_memory, location_window=location_window,
                                solid_overlay=solid_overlay,simple_prompt_ablation=simple_prompt_ablation)
                    
                    answers.append(answer_bad)
                    labels.append(0.0)
                    outputs[bad_file]=answer_bad

                    print('Answer:',answer_bad,'Label:',0.0,'Correct:',answer_bad==0.0)
                    if csv_file is not None:
                        row = [bad_file, answer_bad, 0.0, answer_bad==0.0, organ]
                        # Append the row to the CSV file
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)

                    # Clean up
                    del answer_bad
            torch.cuda.empty_cache()
            gc.collect()

            if limit is not None:
                print('Number of cases:',len(answers))
                if len(answers)>=limit:
                    break

        #calculate accuracy based on answers and labels
        answers=np.array(answers)
        labels=np.array(labels)

        acc=(answers==labels).sum()/(len(answers)-np.where(answers==0.5)[0].shape[0])
        print('Accuracy: ',acc)
        print('Acc:',(answers==labels).sum(),'/(',len(answers),'-',np.where(answers==0.5)[0].shape[0],')')
        print('answers:',answers)
        print('labels:',labels)
        print()

        for k,v in outputs.items():
            print(k,v)



def FewShotErrorDetectionSystematicEvalLMDeploy(pth,n,
                                   base_url='http://0.0.0.0:8000/v1', size=512,
                                    text_region=BodyRegionTextSkeleton, 
                                    organ_descriptions=DescriptionsErrorDetect,
                                    instructions=ZeroShotInstructions,
                                    text_summarize=CompareSummarizeED, organ='liver',
                                    save_memory=True, location_window='skeleton',solid_overlay=False,
                                    anno_window='bone',best=2,
                                    file_list=None,
                                    file_structure='dual',
                                    dice_threshold=0.5,dice_threshold_max=0.9,dice_check=False,
                                    good_examples_path=None,bad_examples_path=None,
                                    limit=None,skip_good=False,skip_bad=False,
                                    csv_file=None,restart=False,dice_list=None,
                                    shapeless=False,min_cases=100,max_cases=10000):
        
        if shapeless:
            organ_descriptions=DescriptionsErrorDetectShapeless

        if solid_overlay=='auto':
            if organ in ['aorta','postcava']:
                solid_overlay=True
            else:
                solid_overlay=False

        if csv_file is not None:
            column_names = ['case', 'answer', 'label', 'correct', 'organ']
            if os.path.exists(csv_file) and restart:
                os.remove(csv_file)
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)
            elif not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)


        #if dice_list is not None:
        #    import pandas as pd
        #    print('Loading dice list:',dice_list)
        #    df=pd.read_csv(dice_list,header=None, names=["case", "dice"])#

            #change dice threshold according to mean and std of dice in list
        #    a=df['dice'].mean()-df['dice'].std()
        #    #order dice in ascending order and get the 100th value
        #    b=df['dice'].sort_values().iloc[100]
        #    dice_threshold=max(dice_threshold,min(a,b))
        #    dice_threshold=min(dice_threshold,dice_threshold_max)
        #    print('Dice threshold re-defined to:',dice_threshold)

        #    filtered_df = df[df["dice"] < dice_threshold]
        #    # Get the list of 'case' values where 'dice' is below the threshold
        #    cases_below_threshold = filtered_df["case"].tolist()
        #    print('Number of cases below threshold:',len(cases_below_threshold))
        #    print('Cases below threshold:',cases_below_threshold)
            
        if dice_list is not None:
            import pandas as pd
            print('Loading dice list:',dice_list)
            df=pd.read_csv(dice_list,header=None, names=["case", "dice"])

            #change dice threshold according to mean and std of dice in list
            #a=df['dice'].mean()-df['dice'].std()
            #order dice in ascending order and get the 100th value
            #b=df['dice'].sort_values().iloc[100]
            #dice_th=max(dice_th,min(a,b))
            dice_th=min(df['dice'].mean()-df['dice'].std(),dice_threshold_max)
            print('Dice threshold re-defined to:',dice_th)

            filtered_df = df[df["dice"] < dice_th]
            # Get the list of 'case' values where 'dice' is below the threshold
            cases_below_threshold = filtered_df["case"].tolist()
            cases_high_dice=df[df["dice"] >= 0.95]["case"].tolist()
            if len(cases_below_threshold)<min_cases:
                #if too few cases, get 100 lowest dice cases to compare
                dice_th=df['dice'].sort_values().iloc[min_cases]
                #dice_th=min(dice_th,dice_threshold_max)
                #filtered_df = df[df["dice"] < dice_th]
                # Get the list of 'case' values where 'dice' is below the threshold
                cases_below_threshold = filtered_df["case"].tolist()
            if len( cases_below_threshold)>max_cases:
                #limit to 10% of dataset
                dice_th=df['dice'].sort_values().iloc[500]
                filtered_df = df[df["dice"] < dice_th]
                # Get the list of 'case' values where 'dice' is below the threshold
                cases_below_threshold = filtered_df["case"].tolist()

        answers=[]
        labels=[]
        outputs={}
        
        files_good,files_bad=get_files(pth,file_list,anno_window,organ,location_window,best,file_structure=file_structure)
        #print('File list:',file_list)
        #sort files
        files_good.sort()
        files_bad.sort()

        if good_examples_path is None:
            files_good_ex=files_good
        else:
            files_good_ex,_=get_files(good_examples_path,file_list,anno_window,organ,location_window,best,file_structure='pick_good_only')

        if bad_examples_path is None:
            files_bad_ex=files_bad
        else:
            _,files_bad_ex=get_files(bad_examples_path,file_list,anno_window,organ,location_window,best,file_structure='pick_bad_only')

        print('Good examples:',files_good_ex)
        print('Bad examples:',files_bad_ex)
        
        #if dice_check:
        #    tmp_good=[]
        #    tmp_bad=[]
        #    for i in range(len(files_good)):
        #        good_file=files_good[i][1]
        #        bad_file=files_bad[i][1]
        #        dice=check_dice(good_file,bad_file)
        #        if dice_list is not None:
        #            raise ValueError('Implement dice check using dice list')
        #        if dice<=dice_threshold:
        #            tmp_good.append(files_good[i])
        #            tmp_bad.append(files_bad[i])
        #    print('Dice check completed. Number of original files:',len(files_good)+len(files_bad))
        #    print('Number of files after dice check:',len(tmp_good)+len(tmp_bad))
        #    files_good=tmp_good
        #    files_bad=tmp_bad

        for i in list(range(max(len(files_good),len(files_bad)))):
            skip_good=False
            skip_bad=False
            if dice_check:
                if dice_list is None:
                    good_file=files_good[i][1]
                    bad_file=files_bad[i][1]
                    dice=check_dice(good_file,bad_file)
                    high_dice=dice>dice_threshold
                    print('Dice:',dice)
                else:
                    clean=files_good[i][0]
                    if '/' in clean:
                        high_dice=clean[clean.rfind('/')+1:].replace('ct_window_skeleton','ct_window_bone') not in cases_below_threshold
                    else:
                        high_dice=clean.replace('ct_window_skeleton','ct_window_bone') not in cases_below_threshold
                
                if high_dice:
                    #print('Dice above th, skipping')
                    continue
            good_examples = files_good_ex[:i] + files_good_ex[i+1:]
            print(len(good_examples))
            good_examples = random.sample(good_examples, max(1,n//2))
            good_examples = [x[1] for x in good_examples]
            bad_examples= files_bad_ex[:i] + files_bad_ex[i+1:]
            bad_examples = random.sample(bad_examples, n//2)
            bad_examples = [x[1] for x in bad_examples]

            #print('Good examples:',good_examples)
            #print('Bad examples:',bad_examples)

            if i<len(files_good):
                clean,good_file=files_good[i]
                if ((csv_file is not None) and (not restart)):
                    if check_case_exists(csv_file, good_file):
                        print('Already exists, skipping case ',good_file)
                        skip_good=True
                if not skip_good:
                    print('Case below:',good_file)
                    answer_good=ErrorDetectionLMDeployFewShot(clean=clean, y=good_file,
                                good_examples=good_examples, bad_examples=bad_examples,
                                base_url=base_url, size=size,
                                text_region=text_region, 
                                organ_descriptions=organ_descriptions,
                                text_summarize=text_summarize, organ=organ,
                                save_memory=save_memory, location_window=location_window,
                                solid_overlay=solid_overlay,
                                shapeless=shapeless)
                    
                    answers.append(answer_good)
                    labels.append(1.0)
                    outputs[good_file]=answer_good

                    print('Answer:',answer_good,'Label:',1.0,'Correct:',answer_good==1.0)
                    if csv_file is not None:
                        row = [good_file, answer_good, 1.0, answer_good==1.0, organ]
                        # Append the row to the CSV file
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)
                    del answer_good
            
            if i<len(files_bad):
                clean,bad_file=files_bad[i]
                if ((csv_file is not None) and (not restart)):
                    if check_case_exists(csv_file, bad_file):
                        print('Already exists, skipping case ',good_file)
                        skip_bad=True
                if not skip_bad:
                    print('Case below:',bad_file)
                    answer_bad=ErrorDetectionLMDeployFewShot(clean=clean, y=bad_file, 
                                good_examples=good_examples, bad_examples=bad_examples,
                                base_url=base_url, size=size,
                                text_region=text_region, 
                                organ_descriptions=organ_descriptions,
                                text_summarize=text_summarize, organ=organ,
                                save_memory=save_memory, location_window=location_window,
                                solid_overlay=solid_overlay)
                    
                    answers.append(answer_bad)
                    labels.append(0.0)
                    outputs[bad_file]=answer_bad

                    print('Answer:',answer_bad,'Label:',0.0,'Correct:',answer_bad==0.0)
                    if csv_file is not None:
                        row = [bad_file, answer_bad, 0.0, answer_bad==0.0, organ]
                        # Append the row to the CSV file
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)
                    # Clean up
                    del answer_bad
            torch.cuda.empty_cache()
            gc.collect()

            if limit is not None:
                print('Number of cases:',len(answers))
                if len(answers)>=limit:
                    break

        #calculate accuracy based on answers and labels
        answers=np.array(answers)
        labels=np.array(labels)

        acc=(answers==labels).sum()/(len(answers)-np.where(answers==0.5)[0].shape[0])
        print('Accuracy: ',acc)
        print('Acc:',(answers==labels).sum(),'/(',len(answers),'-',np.where(answers==0.5)[0].shape[0],')')
        print('answers:',answers)
        print('labels:',labels)
        print()

        for k,v in outputs.items():
            print(k,v)


CompareInstructionsZeroShotFirst=("The images I am sending are a frontal projections of a CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. "
                      "The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, the red overlays may correctly "
                      "or incorrectly mark the %(organ)s. I will send 2 two images, and I want you to "
                      "compare them and tell me which on, Image 1 or Image 2, has a better overlay for the %(organ)s. "
                      " Remember that both overlays may have mistakes, but you must select the one that is closer to the"
                    " expected shape and location of the %(organ)s. Here, I am sending you 'Image 1' only. ")

CompareInstructionsZeroShotSecond=("I am sending you now 'Image 2' for comparison. "
                                   "Compare the overlays in the two images and say if Image 1 "
                                   "or Image 2 has a better overlay for the %(organ)s. "
                                   "Justify your answer very well.")

CompareInstructionsZeroShotUnified=("The images I am sending are a frontal projections of a CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. "
                      "The left side of the image represents the right side of the human body. "
                      "The image may not present the whole human body. But, if the %(organ)s is present in the images, it should "
                     "be marked in red, using an overlay. However, the red overlays may correctly "
                      "or incorrectly mark the %(organ)s. I am sending here 2 two images, and I want you to "
                      "compare them and tell me which of them, Image 1 or Image 2, has a better overlay for the %(organ)s. "
                      " Remember that both overlays may have mistakes, but you must select the one that is closer to the"
                    " expected shape and location of the %(organ)s. Or, in case the %(organ)s should not be present in the image, "
                    "you select the image with as little red as possible. ")


CompareAdditionalInstructionsLiver=(" When evaluating and comparing the annotaions, "
                                    "consider the following anatomical information: the liver is a large,"
                                    " triangular organ located in the upper right quadrant of the abdomen (left in the image), "
                                    "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
                                    " into the left upper quadrant. "
                                    "The liver position is primarily under the rib cage. "
                                    "If the CT scan does not include the liver region (e.g., a scan showing just"
                                    "the pelvis), the correct image is the one with no red overlay."
                                    "Therefore, images with no red region may represent the correct overlay.\n")

CompareSummarizeInstructions=("The text below represents a comparisons of 2 images, 'Image 1' and 'Image 2'. "
                              "The images are frontal projections of a CT scan, with transparency, like an X-ray. "
                              "A red overlay marks the %(organ)s region in the images. "
                              "The comparison in the text evaluates which image has a better overlay for the %(organ)s. "
                              "According to the text, which image has the better overlay for the %(organ)s? "
                              "In other words, which image is better?"
                              "Answer in 2 words only, 'Image 1' or 'Image 2'.\n"
                              "The text you should analzye is:\n")

CompareSummarizeInstructionsRepeatImage=("Summarize your last answer, which image has the better overlay for the %(organ)s? "
                                        "In other words, which image is better?"
                                        "Answer in 2 words only, 'Image 1' or 'Image 2'.")

InstructionList=("Answer each of these questions for me, in order: \n"
                "1- Which region of the body does the image represent? \n"
                "2- In which region of the human body is the %(organ)s usually present? \n"
                "3- Is the region you answered for question 2 part of the region you answered for question 1? The abdomen is NOT part of the pelvis.  \n"
                "4- Which image has the least ammount of red overlay (or no red colors at all)? \n"
                "If you answered no to question 3, conclude that the correct image is the one with no red overlay, or with the smallest red area, i.e., your answer to question 4, and do not answer the remaining questions. \n"
                "5- Considering your answer to question 1, in which region of the image it should the %(organ)s be (the right side of the image is the left side of the human body)? \n"
                "6- Consider the region you asnwered in question 5, in which image do you see red in this region? Image 1, Image 2, both images, or none? \n"
                "If you answered 'Image 1' to question 6, conclude that the correct image is Image 1, and do not answer the remaining questions. \n"
                "If you answered 'Image 2' to question 6, conclude that the correct image is Image 2, and do not answer the remaining questions. \n"
                "7- In which image is the red overlay shape more similar to the expected shape of the %(organ)s? \n"
                "If you arrived at this point and did not stop earlier (after question 4 or 6), your final answer is your answer to question 7.")





def Compare2AnnotationsZeroShot(img1,img2,model,processor,
                                text1=CompareInstructionsZeroShotFirst,
                                text2=CompareInstructionsZeroShotSecond,
                                organ='liver',size=None,prt=True):
    text1=text1%{'organ':organ}
    text2=text2%{'organ':organ}

    if isinstance(img1, str):
        img1 = Image.open(img1)
    if size is not None:
        img1=resize_image(img1,size)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if size is not None:
        img2=resize_image(img2,size)

    if prt:
        print('Image 1:')
        display(img1)
        print('Image 2:')
        display(img2)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text1}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", 
                 "text": "Thank you for the first image, please send me the second on for comparison."},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text2}
                ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(prompt)
    inputs = processor(text=prompt, images=[img1,img2],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 2000, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text

def Compare2AnnotationsZeroShotSinglePrompt(img1,img2,model,processor,
                                text=CompareInstructionsZeroShotUnified,
                                organ='liver',size=None,prt=True):
    text=text%{'organ':organ}

    if isinstance(img1, str):
        img1 = Image.open(img1)
    if size is not None:
        img1=resize_image(img1,size)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if size is not None:
        img2=resize_image(img2,size)

    if prt:
        print('Image 1:')
        display(img1)
        print('Image 2:')
        display(img2)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": text}
                ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img1,img2],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 400, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text


    
def Compare2AnnotationsZeroShot2StepsSinglePrompt(  img1,img2,processor,model,
                                                    text=CompareInstructionsZeroShotUnified,
                                                    textSummary=CompareSummarizeInstructions,
                                                    organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShotSinglePrompt(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text=text,
                        organ=organ,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": textSummary+model_text}
                ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, 
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5

    
def Compare2AnnotationsZeroShot2Steps(  img1,img2,processor,model,
                                        text1=CompareInstructionsZeroShotFirst,
                                        text2=CompareInstructionsZeroShotSecond,
                                        textSummary=CompareSummarizeInstructions,
                                        organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShot(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2,
                        organ=organ,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textSummary+model_text}
                ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt,return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5

def Compare2AnnotationsZeroShot2StepsLarge(  img1,img2,processor,model,
                                        text1=CompareInstructionsZeroShotFirst,
                                        text2=CompareInstructionsZeroShotSecond,
                                        textSummary=CompareSummarizeInstructionsRepeatImage,
                                        organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShot(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2,
                        organ=organ,size=size)

    text1=text1%{'organ':organ}
    text2=text2%{'organ':organ}
    if isinstance(img1, str):
        img1 = Image.open(img1)
    if size is not None:
        img1=resize_image(img1,size)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if size is not None:
        img2=resize_image(img2,size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text1}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", 
                 "text": "Thank you for the first image, please send me the second on for comparison."},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text2}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": model_text}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textSummary},
            ],
    },
]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img1,img2],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5
    
def Compare2AnnotationsZeroShot2Steps(  img1,img2,processor,model,
                                        text1=CompareInstructionsZeroShotFirst,
                                        text2=CompareInstructionsZeroShotSecond,
                                        textSummary=CompareSummarizeInstructions,
                                        organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShot(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2,
                        organ=organ,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": textSummary+model_text}
                ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, 
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5
    
def SystematicComparison(good_annos,bad_annos,model,processor,
                          text1=CompareInstructionsZeroShotFirst,
                          text2=CompareInstructionsZeroShotSecond,
                          organ='liver',size=None):
    answers=[]
    for i,target in enumerate(good_annos,0):
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        good=random.randint(1,2)
        if good==1:
            answer=Compare2AnnotationsZeroShot2Steps(
                            img1=good_annos[i],img2=bad_annos[i],
                            model=model,processor=processor,
                            text1=text1,
                            text2=text2,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 1')
            if answer==1:
                answers.append(1)
            elif answer==2:
                answers.append(0)
        else:
            answer=Compare2AnnotationsZeroShot2Steps(
                            img1=bad_annos[i],img2=good_annos[i],
                            model=model,processor=processor,
                            text1=text1,
                            text2=text2,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers),')')

def SystematicComparisonSinglePrompt(good_annos,bad_annos,model,processor,
                                                    text=CompareInstructionsZeroShotUnified,
                          organ='liver',size=None):
    answers=[]
    for i,target in enumerate(good_annos,0):
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        good=random.randint(1,2)
        if good==1:
            answer=Compare2AnnotationsZeroShot2StepsSinglePrompt(
                            img1=good_annos[i],img2=bad_annos[i],
                            model=model,processor=processor,
                            text=text,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 1')
            if answer==1:
                answers.append(1)
            elif answer==2:
                answers.append(0)
        else:
            answer=Compare2AnnotationsZeroShot2StepsSinglePrompt(
                            img1=bad_annos[i],img2=good_annos[i],
                            model=model,processor=processor,
                            text=text,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
        
        
        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers),')')



def SendMessageQwen(img_file_list, model, processor,  process_vision_info,text, conversation=[],size=None,
                prt=True):
        
    conversation.append({"role": "user","content": [{"type": "text", "text": text}]})
    if len(img_file_list) > 0:
        for i,img in enumerate(img_file_list,0):
            conversation[-1]["content"].append({"type": "image","image":img})
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                           tokenize=False)
    
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generate_kwargs = {"max_new_tokens": 200, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    conversation.append({"role": "assistant","content": [{"type": "text", "text": answer}]})

    if prt:
        for i in range(len(img_file_list)):
            display(Image.open(img_file_list[i]))
        print('Text:',text)
        print('Answer:',answer)

    return conversation, answer

def resize_and_encode_image(image_path, size=512, solid_overlay=False):
    # Open the image using PIL
    print(image_path)
    with Image.open(image_path) as img:
        # Get the original width and height of the image
        original_width, original_height = img.size

        # Determine the scaling factor to adjust the largest side to the desired size
        if original_width > original_height:
            # Width is the largest side
            new_width = size
            new_height = int((original_height / original_width) * size)
        else:
            # Height is the largest side
            new_height = size
            new_width = int((original_width / original_height) * size)

        # Resize the image while maintaining aspect ratio
        img = img.resize((new_width, new_height))

        if solid_overlay:
            img=SolidOverlay(img)

        # Save the resized image to a temporary buffer
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        print('Image resized to:', img.size)

        # Encode the image as base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
def encode_image(image_path,solid_overlay=False):
  with open(image_path, "rb") as image_file:
    if solid_overlay:
        img=SolidOverlay(Image.open(image_path))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        return base64.b64encode(image_file.read()).decode('utf-8')


def truncate_base64(data, max_length=30):
    if len(data) > max_length:
        return f"{data[:max_length]}... (truncated, {len(data)} chars)"
    return data

def get_image_size_from_base64(base64_string):
    # Remove the "data:image/png;base64," prefix if present
    base64_data = base64_string.split(",")[1] if "," in base64_string else base64_string
    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_data)
    # Load image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Create a figure with no margins or padding
    fig = plt.figure(frameon=False)

    # Set the figure size to match the image size
    fig.set_size_inches(image.width / fig.dpi, image.height / fig.dpi)

    # Add the image to the figure without axes, borders, or whitespace
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)

    # Show the image
    plt.show()

    # Check the pixel value at the top-left corner
    print('Top pixel:', image.getpixel((0, 0)))

    # Get image size (width, height)
    img_size = image.size
    # Get file size in bytes
    img_file_size = len(image_data)

    return img_size, img_file_size

clt=None
mdl=None

def InitializeOpenAIClient(base_url='http://0.0.0.0:8000/v1'):
    from openai import OpenAI
    global clt, mdl
    if clt is not None:
        return clt,mdl
    else:
        # Initialize the client with the API key and base URL
        clt = OpenAI(api_key='YOUR_API_KEY', base_url=base_url)

        # Define the model name and the image path
        mdl = clt.models.list().data[0].id# Update this with the actual path to your PNG image
        print('Initialized model and client.')
        return clt,mdl

def CreateConversation(img_file_list, text, conver,size=None,prt=True,solid_overlay=False,role='user'):
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    cnv=copy.deepcopy(conver)
    
    cnv.append({
            'role': role,
            'content': [{
                'type': 'text',
                'text': text,
            }],
        })
    
    imgs=[]
    print('img_file_list:',img_file_list)
    for i,img in enumerate(img_file_list,0):
        if isinstance(size, list):
            s=size[i]
        else:
            s=size
        if s!=None:
            img = resize_and_encode_image(img, s, solid_overlay=solid_overlay)
        else:
            img = encode_image(img, solid_overlay=solid_overlay)
        cnv[-1]['content'].append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{img}"
                                            },
                                        })
        if prt:
            image_size, file_size = get_image_size_from_base64(f"data:image/png;base64,{img}")
            print(f"Image Size (WxH) in prompt: {image_size}, File Size: {file_size} bytes")
            #print(text)
    return cnv

def print_conv(conver):
    # Print the conversation with truncated base64 data
    for entry in conver:
        print(f"Role: {entry['role']}")
        for content in entry['content']:
            if content['type'] == 'text':
                print(f"Text: {content['text']}")
            elif content['type'] == 'image_url':
                image_url = content['image_url']['url']
                truncated_url = truncate_base64(image_url)
                # Extract image size from the base64 string
    i=0
    for entry in conver:
        for content in entry['content']:
            if content['type'] == 'image_url':
                i+=1
    print('Number of images in the conversation:',i)

def request_VLM(cv,model_name,client,max_tokens):
    if max_tokens is None:
        return client.chat.completions.create(
            model=model_name,
            messages=cv,
            temperature=0,
            top_p=1,
            timeout=6000)
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=cv,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            timeout=6000)

def SendMessageLmdeploy(img_file_list, text, conver, base_url='http://0.0.0.0:8000/v1',  
                        size=None,prt=True,print_conversation=False,max_tokens=None,
                        solid_overlay=False,
                        batch=1):
    """
    Sends a message to the LM deploy API.

    Args:
        img_file_list (list): A list of image file paths.
        text (str): The text message to send.
        conver (list): A list of conversation objects.
        base_url (str, optional): The base URL of the LM deploy API. Defaults to 'http://0.0.0.0:8000/v1'.
        size (int, optional): The size to resize the images to. Defaults to None.
        prt (bool, optional): Whether to print the images and conversation. Defaults to True.
        print_conversation (bool, optional): Whether to print the conversation. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens in the completion response. Defaults to None.

    Returns:
        tuple: A tuple containing the updated conversation and the answer from the LM deploy API.
    """
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    client,model_name=InitializeOpenAIClient(base_url)

    if batch>1:
        for i in range(batch):
            #print('Batch:',i)
            #print('img_file_list:',img_file_list[i])
            #print('text:',text[i])
            #print('conver:',conver[i])
            conver[i]=CreateConversation(img_file_list=img_file_list[i], text=text[i], conver=conver[i],size=size,solid_overlay=solid_overlay,prt=(i==0))
            if print_conversation:
                print_conv(conver[i])
    else:
        conver=CreateConversation(img_file_list=img_file_list, text=text, conver=conver,size=size,solid_overlay=solid_overlay)
        if print_conversation:
            print_conv(conver)

    response=[]
    for i in range(batch):
        if batch==1:
            response=request_VLM(conver,model_name,client,max_tokens)
        else:
            # Use ThreadPoolExecutor to send both requests concurrently
            with ThreadPoolExecutor() as executor:
                # Map both the conversation and model name to each thread
                response = list(executor.map(request_VLM, conver, [model_name] * len(conver),[client] * len(conver),[max_tokens] * len(conver)))
        

    if batch==1:
        # Print the response
        answer = response.choices[0].message.content
        conver.append({"role": "assistant","content": [{"type": "text", "text": response.choices[0].message.content}]})
        if prt:
            print('Text:',text)
            print('Answer:',answer)
    else:
        answer=[]
        for i in range(batch):
            answer.append(response[i].choices[0].message.content)
            conver[i].append({"role": "assistant","content": [{"type": "text", "text": response[i].choices[0].message.content}]})
        if prt:
            print('Text:',text[0])
            print('Answer:',answer[0])

    return conver, answer

BodyRegionTextV0=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It looks like an AP (anterior-to-posterior) X-ray. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and where the image limits are. Present a complete list of all organs usually present in this body region (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q2 = _\n")

BodyRegionTextV1=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It looks like an AP (anterior-to-posterior) X-ray. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and which organs inside this region are touching the image's upper boundary and its lower boundary. Present a complete list of all organs usually present in the body region contained in the image (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q2 = _\n")
#seem to give maybe more accurate answers


#touching became crossing

BodyRegionTextHighlightedSkeleton=("The image I am sending is frontal projections of one CT scan. "
                "Is it not a CT slices but a transparent view of the entire body, similar to an X-ray that allows a see-through visualization of internal structures. It looks like an AP (anterior-to-posterior) X-ray."
                "The image focuses on the bones. \n"
"Q1- Look at the image carefully, tell me very precisely where in the body the image lower and upper limits are. Present a complete list of all organs usually present in the body area within the image (just list their names).\n"
"Q2- Which of these organs are partially contained in the image? In other words, which organs touch the image uper or lower limits.\n"
"Q3- Based on your answer to Q1, is the %(organ)s usually present (even if partially) in this area and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q3 = _\n")

BodyRegionTextSkeletonGoodOne=("The image I am sending is frontal projections of one CT scan, focusing on showing the skeleton. Look at it carefully, and answer the questions below:\n"
"Q1- Which bones are on the top of the image? Bones are on its bottom?\n"
"Q2- Which of the following landmarks are present in the image? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"skull = _"
"neck = _"
"trachea = _"
"upper ribs = _"
"lower ribs = _"
"lumbar spine = _"
"pelvis = _"
"femurs = _"
"Q3- Considering these landmarks and the bones on the image top and bottom, give me a complete list of all organs (not bones) usually contained within this image limits (just list their names).\n"
"Q4- Based on your answer to Q2 and Q3, is the %(organ)s usually present within this image limits? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"Q4 = _\n")


BodyRegionTextSkeletonV0=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It highlights major bones, not soft tissue. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and where the image limits are. Present a complete list of all organs usually present in this body region (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"Q2 = _\n")


BodyRegionTextSkeletonV1 = ("I am sending you a figure with 2 images inside it, Image 1 (left) and Image 2 (right). The images are two frontal projections from the same CT scan. "
"They are not CT slices but a transparent view of the entire body, similar to an X-ray that allows a see-through visualization of internal structures. They look like AP (anterior-to-posterior) X-rays."
"Image 1 uses a window that displays soft tissue, while Image 2 focuses on the skeleton, but they show exactly the same body region. "
"Please carefully analyze the image and follow the steps below to ensure accuracy:\n\n"
"Step 1: Describe exactly which parts of the body are visible in the image, including where the image starts and ends. Focus on the visible boundaries, and if the image is cropped, mention where the cropping occurs. "
"Do not include parts of the body that are not visible in the image.\n\n"
"Step 2: Based on the visible region described in Step 1, provide a list of organs that are actually visible within the image boundaries. "
"Only include organs that are fully or partially visible in the image, and avoid general assumptions about what should be present.\n\n"
"Step 3: Based on your findings from Step 1 and Step 2, is the %(organ)s visible in the image? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by 'Yes' or 'No':\n"
"Step 3 = _\n")

BodyRegionTextBad=("The image I am sending is a frontal projection of one CT scan. It is not a CT slice but a projection similar to an X-ray, allowing you to see through the body. Please answer the following questions carefully:\n"
                "Q1- Identify the body region shown in the image. Specifically, indicate if it includes the **upper chest**, **mid chest**, **lower chest**, **abdominal regions** and **pelvic regions**. List all organs that are typically found in the body region(s) you identify, including whether the **trachea** is normally visible in this region.\n"
                "Q2- Based on your answer to Q1, is the %(organ)s normally present in this body region? Provide a 'yes' or 'no' answer using the template below, substituting _ with Yes or No.:\n"
                "Q2 = _\n")


AorticArchTextSkeletonV0=("If you answered no to Step 3, skip Step 4. If you answered yes, continue:\n"
                "Step 4: Are heart and aoric arch fully contained within image bondaries? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by 'yes' or 'no':\n"
                "Step 4 = _\n")


AorticArchTextV5=("If you answered no to Q2, skip Q3 and Q4. If you answered yes, continue:\n"
                "Q3- What parts of the respiratory system are usually presentusually present in this region and in your list?\n"
                "Q4- Based on your answer to Q3, is the trachea usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
                "Q4 = _\n")
#80% accuracy but there is a better one below!

AorticArchTextV4=("If you answered no to Q2, skip Q3 and Q4. If you answered yes, continue:\n"
                "Q3- The **trachea** is part of the respiratory system, and it is located **only in the upper chest**, near the midline of the body. It appears as a **vertical air-filled tube**, running from the neck down to the top of the lungs and connecting to the bronchi. Importantly, if **upper chest** organs and the **top of the lungs** are not visible in the image, the trachea cannot be present.\n"
                "Look at the image and determine if the **upper chest** is visible, specifically focusing on the area above the lungs. If only the lower parts of the lungs are visible, the trachea should not be visible. Identify which parts of the respiratory system are visible.\n"
                "Q4- Based on your answer to Q3, is the **trachea** visible in this image? Answer 'yes' or 'no' using the template below, substituting _ with Yes or No. Be sure to consider that the trachea cannot be present without the **upper chest** being visible, and it should appear as a distinct air-filled tube above the lungs:\n"
                "Q4 = _\n")

AorticArchTextV3=("If you answered no to Q2, skip Q3 and Q4. If you answered yes, continue:\n"
                "Q3- Look closely at the image. The **trachea** is part of the respiratory system and should appear as a **vertical air-filled tube in the upper chest**, located in the midline, above the lungs. It is positioned just below the neck and above the heart. If only the lower parts of the lungs are visible, the trachea is unlikely to be present. Based on this information, identify which parts of the respiratory system (if any) are visible in this image.\n"
                "Q4- Based on your answer to Q3, is the **trachea** visible in this image? Answer 'yes' or 'no' using the template below, substituting _ with Yes or No. Be sure to consider whether the upper chest is visible and whether the trachea’s position is above the lungs in this image:\n"
                "Q4 = _\n")
#bad

AorticArchTextV2=("Q3- If you answered no to Q2, skip this question (Q3). If you answered yes, continue:\n"
                "In this image, you should identify whether the **upper part of the lungs** is visible, not just the base. The **upper lungs** are located near the top of the ribcage.\n"
                "Pay attention to this distinction: If **only the base of the lungs** is visible, the aortic arch is not in the image.\n"
                "Remember, the **aortic arch** is located in the **upper chest**, near the top of the lungs. Therefore, the presence of the aortic arch depends on whether the upper portion of the lungs is visible in the scan.\n"
                "Now answer this: Are the **upper parts of the lungs** fully visible (not just the base)? If yes, the aortic arch should be visible.\n"
                "Based on your observation, fill in the template below by substituting _ with Yes or No:\n"
                "Q3 = _\n"
                "Please explain briefly whether the upper lungs are visible and why this leads to the conclusion about the aortic arch's presence.")

AorticArchTextV1=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, answer whis new question: the lungs should be two large dark regions in the image. Are the lungs present and fully visible, from top to bottom, in the image? Think carefully, and provide a complete answer with justification. \n"
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")

AorticArchTextV0=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, describe the image's upper limit: does in include the aortic arch? The aortic arch is placed just above the heart, and it will be present if the top of the lungs are visible in the image. Think carefully, and provide a complete answer with justification.\n"
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")

AorticArchTextV6=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, describe the image's upper limit: does it include the aortic arch? The aortic arch is placed just above the heart, and it will be presnt if the top of the lungs are visible in the image. Think carefully, and provide a complete answer with justification.\n"
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")
#80% accuracy

AorticArchTextV7=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, describe the image's upper limit: does it include the aortic arch? The aortic arch is placed just above the heart.\n"
                "Carefully analyze the image and your answer to Q1 to respond Q3, justifying your answer. Moreover, consider the following tips:\n"
                "- The aortic arch is presnt only if the top of the lungs and the trachea are present in the image.\n" 
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")



ComparisonText=("I am now sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Here is the description of each image (image titles are above the images):\n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI), tuned for better bone visualization. The image is superposed with %(organ)s overlay 1 (in red).\n"
"Image 2: frontal projection of a CT scan using a less wide window (400 UI), tuned for better abdominal organ visualization. The image is also superposed with %(organ)s overlay 1 (like image 1).\n"
"Image 3: frontal projection of the CT scan using a wide window (2000 UI). The image is superposed with %(organ)s overlay 2 (in red).\n"
"Image 4: frontal projection of a CT scan using a less wide window (400 UI). The image is also superposed with %(organ)s overlay 2 (like image 3).\n"
"Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonTextContinued=("I am now sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Here is the description of each image (image titles are above the images):\n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI), tuned for better bone visualization. The image is superposed with %(organ)s overlay 1 (in red).\n"
"Image 2: frontal projection of a CT scan using a less wide window (400 UI), tuned for better abdominal organ visualization. The image is also superposed with %(organ)s overlay 1 (like image 1).\n"
"Image 3: frontal projection of the CT scan using a wide window (2000 UI). The image is superposed with %(organ)s overlay 2 (in red).\n"
"Image 4: frontal projection of a CT scan using a less wide window (400 UI). The image is also superposed with %(organ)s overlay 2 (like image 3).\n"
"Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Classes=("I am sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible type of overlay error is overlapping the left and right %(organ_singular)s. These undesirable overlaps (if present) are marked in red in each overlay. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Here is the description of each image (image titles are above the images):\n"
"Image 1 (top right): frontal projection of the CT scan using a wide window (2000 UI), tuned for better bone visualization. The image is superposed with %(organ)s overlay 1 \n"
"Image 2 (top left): frontal projection of a CT scan using a less wide window (400 UI), tuned for better abdominal organ visualization. The image is also superposed with %(organ)s overlay 1 (like image 1).\n"
"Image 3 (bottom left): frontal projection of the CT scan using a wide window (2000 UI). The image is superposed with %(organ)s overlay 2 (in red).\n"
"Image 4 (bottom right): frontal projection of a CT scan using a less wide window (400 UI). The image is also superposed with %(organ)s overlay 2 (like image 3).\n"
"Compare overlay 1 (images 1 and 2, top) to overlay 2 (images 3 and 4, bottom) and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText6Figs=("I am now sending you a figure with 6 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked with colored overlays. However, the overlays may correctly or incorrectly mark the %(organ)s. "
                "Images 1 and 4 (column 1, left), show overlay 1 in red. Images 2 and 5 (column 2, middle), show overlay 2 in yellow. Images 3 and 6 (column 3, right), show the superposition of overlay 1 (red) and 2 (yellow), their intersection is in organge. "
                "The only difference from the images in the first row (1, 2 and 3) and in the second row (images 4, 5 and 6) is contrast, the first row more clearly shows the bones."
                "The letter R in blue inside the images represent the right side of the human body. The letter L in blue represents the left side of the human body. \n"
                "Compare overlay 1 (red) to overlay 2 (yellow) and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Figs=("I am sending you a figure that contains two images, Image 1 on the left and Image 2 on the right. "
                     "These images are frontal projections from the same CT scan, displayed with transparency similar to X-rays. "
                     "In these images, a red overlay has been applied to highlight the %(organ)s region. However, the accuracy of the red overlays in marking the %(organ)s may vary between the two images. "
                     "The letters R (blue) and L (green) are visual markers indicating the right and left sides of the human body, respectively. "
                     "Note that the left side of each image corresponds to the right side of the body, and the right side of the image corresponds to the left side of the body, like AP (anterior-to-posterior) X-rays. "
                     "Please compare Overlay 1 (Image 1) to Overlay 2 (Image 2) and determine which overlay provides a more accurate representation of the %(organ)s. "
                     "You will need to assess the overlays based on the following specific criteria related to the %(organ)s in the next part.")#GPT revised

ComparisonText2FigsV0=("I am now sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. "
                     "They are not a CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body. "
                "The left side of each image represents the right side of the human body. The right side of each image represents the left side of the human body. "
                "Compare overlay 1 (shown in Image 1) to overlay 2 (shown in Images 2) and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText2FigsContinued=("I am sending you a figure with 2 images inside of it. The one on the left, Image 1, is the first image I sent you in this conversation, and the one on the right, Image 2, is the second one. " 
                            " As I previously explained, they are frontal projections of the same CT scan. "
                     "They are not a CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body. "
                "The left side of each image represents the right side of the human body. The right side of each image represents the left side of the human body. "
                "Compare overlay 1, shown in Image 1, to overlay 2, shown in Images 2, and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText2Classes2Figs=("I am sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. "
                             " They are not a CT slices, they have transparency and let you see through the entire human body, like an X-ray. "
                             "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body, which is NOT the right and left sides of the images. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible error is the absence of one of the %(organ)s, in this case one of the collors will be missing. "
                "Another one is marking the left %(organ_singular)s in the right side of the human body or vice-versa. "
                "In these cases, it is possible to observe an undesired overlap between the left and right %(organ_singular)s, any overlap is marked in red. If one %(organ_singular)s is wrongly marked totally inside the other, we will have only red and blue or red and green overlays."
                "You may also encounter errors in the %(organ)s shape and position."
"Your task: Compare overlay 1 in image 1 to overlay 2 in image 2 and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Classes2FigsV2=("I am sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. "
                             " They are not a CT slices, they have transparency and let you see through the entire human body, like an X-ray. "
                             "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body, which is NOT the right and left sides of the images. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible error is the absence of one of the %(organ)s. Another one is marking the left %(organ_singular)s in the right side of the human body or vice-versa. "
                "In these cases, it is possible to observe an undesired overlap between the left and right %(organ_singular)s, any overlap is marked in red. You may also encounter errors in the %(organ)s shape and position."
"Your task: Compare overlay 1 in image 1 to overlay 2 in image 2 and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Classes2FigsV1=("I am sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. They are not a CT slices, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible type of overlay error is overlapping the left and right %(organ_singular)s. These undesirable overlaps (if present) are marked in red in each overlay. "
                "A possible label error is when the left %(organ_singular)s is marked in the right side of the body or vice versa. In these cases, there may be overlaps be"
                "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body (which is not the right and left sides of the images). "
"Your task: Compare overlay 1 in image 1 to overlay 2 in image 2 and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText1Fig=("I am now sending you the same frontal projection of a CT as before, but now with 2 overlays. Overlay 1 is red, overlay 2 is yellow, and their superposition is orange."
                "The overlays try to mask the %(organ)s region in the image. However, the red or the yellow overlay may mark it better %(organ)s. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Although the images represent the same CT scan, the overlays are different. \n"
                "Compare overlay 1 (red) to overlay 2 (yellow) and tell me which one is a better overlay for the %(organ)s.\n"
                "Remember to justify your answer, and remember that the orange region is the overlap between the 2 overlays.")

ComparisonText6FigsSimple=("I sending you a figure with 6 images inside of it. They are all frontal projections of the same CT scan. They look like X-rays."
                "The %(organ)s region in the images should be marked with colored overlays. However, the overlays may correctly or incorrectly mark the %(organ)s. "
                "Images 1 and 4 (column 1, left), show overlay 1 in red. Images 2 and 5 (column 2, middle), show overlay 2 in yellow. Images 3 and 6 (column 3, right), show the superposition of overlay 1 (red) and 2 (yellow), their intersection is in organge. "
                "The only difference from the images in the first row (1, 2 and 3) and in the second row (images 4, 5 and 6) is contrast, the first row more clearly shows the bones."
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. \n"
                "Compare overlay 1 (red) to overlay 2 (yellow) and tell me which one is a better overlay for the %(organ)s.\n")

LiverDescription=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
"a) The liver is a large organ, with triangular or wedge-like shape.\n"
"b) The liver is located in the upper right quadrant of the abdomen (right is indicated with a blue R in the figures), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. It spans across the midline, partially extending into the left upper quadrant. Looking at the bones images 1 and 3, you can easily identify if the liver should appear in the images or not. The liver is not in the pelvis.\n"
"c) The liver is a single structure.\n"
"d) The liver position is primarily under the rib cage.\n"
"e) If the CT scan does not include the %(organ)s region (for example, a scan showing just the pelvis), the correct overlay is actually the one showing no red region (or as little as possible).\n")

NoOrganText=("I am now sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, remember you concluded that the %(organ)s should not be present in the images. "
                "Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one has the smallest red area (ideally none).\n")

NoOrganSimple=("I am sending you 2 images, 'Image 1' on the left, and 'Image 2' on the right. Which of them has the least ammount of red color?\n"
             "Answer only 'Image 1' or 'Image 2'.\n")

NoOrganSimple2Classes=("I am sending you 2 images, 'Image 1' on the left, and 'Image 2' on the right. Which of them has the least ammount of red, blue and green colors?\n"
             "Answer only 'Image 1' or 'Image 2'.\n")

CompareSummarize=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                "The overlays were positioned over 4 images, Image 1 and Image 2 showed Overlay 1, and Image 3 and Image 4 showed Overlay 2. "
                "A LVLM like you compared the 2 overlays by analyzing the 4 images. Its answer is the text below."
                "The text explains which overlay is better. I want you to answer me which overlay is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "If the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")


CompareSummarizeInContext=("The text below represents a comparisons of 2 overlays, 'Overlay X' and 'Overlay Y', which were present in Image X and Image Y. "
                "A LVLM like you compared the 2 overlays by analyzing the images. Its answer is the text below."
                "Notice, however, that this VLM received %(n)s pairs of images as examples, but I only care about the Images X and Y from last pair, pair %(o)s. "
                "The text explains which overlay is better. I want you to answer me which overlay is better according to the text. Answer me with only 2 words: 'Overlay X' or 'Overlay Y'. "
                "If the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")

CompareSummarize2Figs=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " Image 1 showed Overlay 1, and Image 2 showed Overlay 2. "
                "A VLM like you compared the 2 overlays by analyzing the 2 images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overlay is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "The text is:\n")

CompareSummarize2Figs=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " Image 1 showed Overlay 1, and Image 2 showed Overlay 2. "
                "A VLM like you compared the 2 overlays by analyzing the 2 images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overlay is better according to the text. Answer me with only these words: 'Overlay 1', 'Overlay 2' or 'Neither'. "
                "The text is:\n")

CompareSummarize6Figs=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " Image 1 and 4 showed Overlay 1 in red, and Images 2 and 5 showed Overlay 2 in yellow. Images 3 and 6 showed the superposition of both overlays. "
                "A LVLM like you compared the 2 overlays by analyzing the 6 images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overlay is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "In case the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")

CompareSummarize1Fig=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " They were placed over one image, overlay 1 was red, overlay 2 was yellow, and their overlap was orange. "
                "A LVLM like you compared the 2 overlays. Its answer is the text below."
                "The text explains which overlay is better. I want you to answer me which overlay is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "If the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")


KidneysDescription=("Answer each one of the following questions before drawing a conclusion of which overlay is better:\n"
                   "a) Consider that a person usually has two kidneys. Does Image 1 show two distinct kidney overlays? Does Image 2 show two distinct kidney overlays? If both show a single kidney, the patient may truly have only one kidney. If only one images shows a single kidney, the image showing two kidneys should be better. \n"
                   "b) Shape: each kidney has a bean-shaped structure, with a convex lateral surface and a concave medial surface. Are the shapes of the 2 kidneys in image 1 correct? Are the shapes in image 2 better, worse or similar? \n"
                   "c) Location: the kidneys are located in the retroperitoneal space, on either side of the spine, at the level of the lower ribs. Are the locations of the 2 kidneys in image 1 correct? Are the kidney locations in image 2 better, worse or similar? \n")

KidneysDescriptionV0=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
                   "a) Number and location: a person usually has two kidneys.\n"
                   "b) Shape: each kidney has a bean-shaped structure, with a convex lateral surface and a concave medial surface. The concave side features an indentation called the renal hilum.\n"
                   "c) Location: the kidneys are located in the retroperitoneal space, on either side of the spine, at the level of the lower ribs.\n")

AdrenalGlandDescription=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
                   "a) Number: a person usually has two adrenal glands, one on top of each kidney.\n"
                   "b) Location: the adrenal glands are located on the superior aspect of each kidney, in the retroperitoneal space.\n"
                   "c) Right Adrenal Gland Shape (right side of the body): triangular shape.\n"
                   "d) Left Adrenal Gland Shape (left side of the body): Generally crescent-shaped or semilunar. May appear as a curved line or elongated structure above the kidney.\n"
                   "e) Size: adrenal glands are relatively small compared to the kidneys.\n")

AortaDescriptionV0=("Answer each one of the following questions before drawing a conclusion of which overlay is better:\n"
                     "a) Shape: In a CT projection, the aorta should be marked as a long vertical red line, with a curve on its top, similar to the shape of a question mark (?). Does the shape of the red region in image 1 match the shape of the aorta? Does the shape of the red region in image 2 match the shape of the aorta? If the shape of the aorta is better represented in one of the images, that image should be better. \n"
                     "b) Location: after the small curve on its top (aortic arch), the aorta should be parallel to the spine. Usually it is a long vertical line in the midline of the body. The aorta can be curved if the spine is also curved. Is the aorta correctly positioned in image 1? Is it correctly positioned in image 2? \n"
                     "c) Completeness: the aorta should be visible from the heart to the pelvis. Is the aorta complete in image 1? Is it complete in image 2? \n")
#Acc: 2/3
AortaDescription = (
    "Analyze the two images to determine which overlay represents the aorta more accurately, based on the following criteria:\n"
    "1. Shape: The aorta should appear as a long vertical red line with a curve at the top, resembling a question mark (?).\n"
    "- In Image 1, does the shape of the red region resemble this description?\n"
    "- In Image 2, does the shape of the red region resemble this description?\n"
    "- Which image has a shape closer to the expected form of the aorta?\n"
    "2. Location: After the small curve at the top (aortic arch), the aorta should run parallel to the spine, which usually appears as a vertical line along the midline of the body.\n"
    "- Is the aorta correctly positioned in Image 1 relative to the spine?\n"
    "- Is the aorta correctly positioned in Image 2 relative to the spine?\n"
    "- Which image shows a better location for the aorta?\n"
    "3. Completeness: The aorta should be visible from the heart down to the pelvis (or down to the image bottom, if the pelvis is not visible).\n"
    "- Is the aorta fully visible and complete in Image 1?\n"
    "- Is the aorta fully visible and complete in Image 2?\n"
    "- Which image shows a more complete representation of the aorta?\n"
    "Based on the shape, location, and completeness, indicate which overlay (Image 1 or Image 2) provides a better representation of the aorta."
)
#Acc: 3/3
DescendingAortaDescriptionV0=("Answer each one of the following questions before drawing a conclusion of which overlay is better:\n"
                     "a) Shape: The aortic arch is not visible in this CT projection, Thus, the aorta should be a long vertical red line (tubular structure), clearly visible at the very top of the image. A common overlay error is not displaying the aorta at the very top of the image. Is the red overlay in Image 2 visible at the very top of the image? Is the red overlay in Image 2 visible at the very top of the image? \n"
                     "b) Location: the aorta should be parallel to the spine. Usually it is a long vertical line in the midline of the body. The aorta can be curved if the spine is also curved. Is the aorta correctly positioned in image 1? Is it correctly positioned in image 2? \n"
                     "c) Completeness: the aorta should be visible from the top of the image to the pelvis (if the pelvis is present in the ct). Is the aorta complete in image 1? Is it complete in image 2? \n")
#Accuracy:  62.5 ( 5 / 8 )

DescendingAortaDescriptionV1=("Answer the following questions before concluding which overlay is a better annotation for the aorta:\n"
                            "a) **Upper Edge**: The aortic arch is not visible in this CT projection. Therefore, the aorta should appear as a long, vertical red line (tubular structure), extending from the **top edge** of the image. A common mistake is the aorta overlay not reaching the very top of the image. Does the red overlay in Image 1 touch the **upper edge** of the image? Does the red overlay in Image 2 touch the **upper edge** of the image? Compare the two images and identify which one, if either, has the red overlay reaching the top edge more accurately.\n"
                            "b) **Location and Shape**: The aorta should be parallel to the spine, typically as a long vertical line in the midline of the body. The aorta may curve if the spine is also curved. Is the aorta correctly positioned in Image 1? Is it correctly positioned in Image 2? Compare both images and explain which overlay follows the correct position better.\n"
                            "c) **Completeness**: The aorta should extend from the top of the image to the pelvis (if the pelvis is visible in the scan). Is the aorta visible from the top edge to the pelvis in Image 1? Is it complete in Image 2? Compare the completeness of both images.")
#Accuracy:  80%, but errors were only in the trachea detection

DescendingAortaDescriptionV2=("Answer the following questions before concluding which overlay is a better annotation for the aorta:\n"
                            "a) **Upper Edge**: The aortic arch is not visible in this CT projection. Therefore, the aorta should appear as a long, vertical red line (tubular structure), extending from the **top edge** of the image. The overlay must display the aorta at the very top of the image. Does the red overlay in Image 1 touch the **upper edge** of the image? Does the red overlay in Image 2 touch the **upper edge** of the image? Compare the two images and identify which one, if either, has the red overlay reaching the top edge more accurately.\n"
                            "b) **Location and Shape**: The aorta should be parallel to the spine, typically as a long vertical line in the midline of the body. The aorta may curve if the spine is also curved. Is the aorta correctly positioned in Image 1? Is it correctly positioned in Image 2? Compare both images and explain which overlay follows the correct position better.\n"
                            "c) **Completeness**: The aorta should extend from the top of the image to the pelvis (if the pelvis is visible in the scan). Is the aorta visible from the top edge to the pelvis in Image 1? Is it complete in Image 2? Compare the completeness of both images.")
#Accuracy:  100%-Seems biased to say image 2!

DescendingAortaDescription=("Answer the following questions before concluding which overlay is a better annotation for the aorta:\n"
                  "a) **Upper Edge**: The aortic arch is not visible in this CT projection, so the aorta should appear as a long, vertical red line (tubular structure), extending from the **top edge** of the image. Compare Image 1 and Image 2: Does either red overlay touch the **upper edge** of the image more accurately? Which overlay better represents the aorta from the top edge of the image?\n"
                  "b) **Location and Shape**: The aorta should be parallel to the spine, typically as a long vertical line in the midline of the body. It may curve if the spine is curved. Compare Image 1 and Image 2: Which red overlay better represents the correct position and shape of the aorta relative to the spine? Is one overlay more accurately following the spine than the other?\n"
                  "c) **Completeness**: The aorta should extend from the **top of the image to the pelvis** (if the pelvis is visible in the scan). Compare Image 1 and Image 2: Which red overlay more accurately shows the full length of the aorta from the top of the image to the pelvis? Which image is more complete?\n")

IVCDescription=("Answer the following questions before concluding which overlay is a better annotation for the Inferior Vena Cava (IVC or postcava):\n"
                "a) **Upper Edge**: The IVC is not visible in the upper chest, but it should appear as a long vertical red line (tubular structure), running **parallel to the aorta** and extending from the mid-abdomen upward to the diaphragm. The overlay must display the IVC near the **top of the abdomen** (but not the very top of the image). Does the red overlay in Image 1 extend close to the top of the abdomen? Does the red overlay in Image 2 reach this region more accurately? Compare the two images and identify which one more accurately represents the upper portion of the IVC.\n"
                "b) **Location and Shape**: The IVC is a vertical structure located slightly to the **right of the spine**, running parallel to the descending aorta. It should appear as a vertical tubular structure, without the curves seen in the aorta. Does the red overlay in Image 1 show the IVC in the correct position relative to the spine and the aorta? Does Image 2 show the IVC correctly positioned? Compare both images and explain which overlay is more accurately positioned.\n"
                "c) **Completeness**: The IVC should extend from the **diaphragm down to the pelvis**, collecting blood from the lower body. The overlay should represent the IVC from its uppermost part in the abdomen (below the heart) to its lowermost part at the pelvis. Does Image 1 show the IVC from the diaphragm to the pelvis? Does Image 2? Compare the completeness of the IVC from top to bottom in both images.")
#created with gpt from the arta description

Descriptions={"aorta":AortaDescription,
              "descending aorta":DescendingAortaDescription,
              "postcava":IVCDescription,
              "liver":LiverDescription,
              "kidneys":KidneysDescription,
              "adrenal_glands":AdrenalGlandDescription}

def RedArea(image_path):
    # Open the image
    image = Image.open(image_path)

    # Ensure the image is in RGB mode
    image = image.convert('RGB')

    # Convert image data to a NumPy array
    data = np.array(image)

    # Separate the red, green, and blue channels
    red_channel = data[:, :, 0]
    green_channel = data[:, :, 1]
    blue_channel = data[:, :, 2]

    # Create a boolean mask where both green and blue channels are less than the red channel
    mask = (green_channel < red_channel) & (blue_channel < red_channel)

    # Calculate the number of pixels where the condition is True
    area = np.sum(mask)

    return area

TextCompare=('Based on your analyses of the two figures, which of the overlays is a better overlay for the %(organ)s, the first one (overlay 1, from Image 1) or the second one (overlay 2, from Image 2)?\n'
             'Consider your previous analyses, but confirm these analyses by analyzing and comparing the two images again. Consider the shape, location, and completeness of the %(organ)s in each overlay. Think thoroughly and justify your answer.')

TextCompareAdd=('First, analyze possible errors in overlay 2, in doing so, directly compare it to overlay 1. In this comparison, you may find that either aspects from overlay 2 are worse than those in overlay 1, or you may find that aspects from overlay 1, which you previously found correct, are actually worse than overlay 2. '
                'After analyzing overlay 2, answer me: which of the overlays is a better overlay for the %(organ)s, the first one (overlay 1, from Image 1) or the second one (overlay 2, from Image 2)?\n'
                'Think thoroughly and justify your answer. If both overlays have mistakes, choose the one whit less mistakes or smaller mistakes. You must always choose one of the overlays, one is for sure better than the other, and they are different. You cannot just say they are both equally good or equally bad, they are not. ')# If you you are sure that both are equally bad or equally good, respond "none".c
TextCompareAdd3Options=('First, analyze possible errors in overlay 2, in doing so, directly compare it to overlay 1. In this comparison, you may find that either aspects from overlay 2 are worse than those in overlay 1, or you may find that aspects from overlay 1, which you previously found correct, are actually worse than overlay 2. '
                'After analyzing overlay 2, answer me: which of the overlays is a better overlay for the %(organ)s, the first one (overlay 1, from Image 1) or the second one (overlay 2, from Image 2)?\n'
                'Think thoroughly and justify your answer, and try your best to truly understand which overlay is better and why. '
                'Say that there is no better overlay only if: you are sure that both are EQUALLY bad / you are sure they are EQUALLY good / both overlays have very obvious mistakes.')
                #'Say that neither overlay is better only in one on these three consitions: you are sure both overlays are equally bad / you are sure both overlays equally good / you cannot be totally sure that an overlay is better than the other.')

TextCompareSuper=('Based on your analyses of the two figures, which of the overlays is a better overlay for the %(organ)s, the first one (overlay 1, from Image 1) or the second one (overlay 2, from Image 2)?\n'
    'Now that you know both images, you may want to revise your previous analyses, as you may find that either aspects from overlay 2 are worse than those in overlay 1, or you may find that aspects from overlay 1, which you previously found correct, are actually worse than overlay 2. '
    'To help you in this comparison, carefully observe the image I am sending now. It shows the same CT scan frontal projection as Image 1 and Image 2, but it shows the 2 overlays. '
    'In this new image, overlay 1 in red and overlay 2 in yellow. The superposition is in orange. '   
    'After throughly analyzing your previous answers and the new image, tell me: which of the overlays is a better overlay for the %(organ)s, the first one (overlay 1, from Image 1) or the second one (overlay 2, from Image 2)?\n'
    'Think thorughly and justify your answer. If both overlays have mistakes, choose the one with less mistakes or smaller mistakes. If you you are sure that both are equally bad or equally good, respond "none".')

def Prompt3MessagesSepFiguresLMDeploy(clean, y1, y2, 
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=DescriptionsED,
                            text_y1=ZeroShotInstructions, 
                            text_y2=ZeroShotInstructions,
                            text_compare=TextCompareAdd,
                            text_summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone',solid_overlay=False):
    
    organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
    if organ=='liver':
        organRegion+=LiverDescriptionLocation
    if organ=='gall_bladder':
        organRegion+=GallbladderDescriptionLocation
    text_compare=text_compare % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    if 'skeleton' in window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
            else:
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
    
    if AnswerNo:
        a1=RedArea(y1)
        a2=RedArea(y2)
        print('Annotation should be zero')
        if a1==0 and a2==0:
            return 0.5
        elif a1==0:
            return 1
        elif a2==0:
            return 2
        else:
            return -1
    #else:
    #    a1=RedArea(y1)
    #    a2=RedArea(y2)
    #    if a1==0 and a2==0:
    #        print('Both annotations are empty, but they should not be')
    #        return 0.5
    #    elif a1==0:
    #        print('y1 is empty, but it should not be')
    #        return 2.0
    #    elif a2==0:
    #        print('y2 is empty, but it should not be')
    #        return 1.0
        
        
    
    
    text_y1 = text_y1 % {'organ': organ.replace('_',' '), 'number': 1} 
    if isinstance(organ_descriptions[organ], list):
        text_y1 += organ_descriptions[organ][0]
    else:
        text_y1 += organ_descriptions[organ]

    text_y2 = text_y2 % {'organ': organ.replace('_',' '), 'number': 2} 
    if isinstance(organ_descriptions[organ], list):
        text_y2 += organ_descriptions[organ][1]
    else:
        text_y2 += organ_descriptions[organ]

    if save_memory:
        conversation=[]

    #Analyze image 1
    imgs=[y1]
    conversation, answer = SendMessageLmdeploy(imgs,text=text_y1, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay)
    
    #Analyze image 2 and compare
    imgs=[y2]
    conversation, answer = SendMessageLmdeploy(imgs,text=text_y2+'\n'+text_compare, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay)
    
    if 'image' not in answer.lower() and 'overlay' not in answer.lower():
        return 0.5
    
    conversation, answer = SendMessageLmdeploy([], text=text_summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5

def CompareAnswers(answer,conservative,in_context=False):
    if in_context:
        o1='overlay x'
        o2='overlay y'
    else:
        o1='overlay 1'
        o2='overlay 2'

    if o1 in answer[0].lower() and o2 not in answer[0].lower():
        a1 = 1
    elif o2 in answer[0].lower() and o1 not in answer[0].lower():
        a1 = 2
    else:
        a1 = 0.5

    if o1 in answer[1].lower() and o2 not in answer[1].lower():
        a2 = 1
    elif o2 in answer[1].lower() and o1 not in answer[1].lower():
        a2 = 2
    else:
        a2 = 0.5

    print("Dual answers are:",a1,a2)

    if conservative:
        if a1==0.5 or a2==0.5:#one of the 2 answers is uncertain: return an uncertain answer
            return 0.5, [a1,a2]
        elif a1==a2:#both answers are the same, but I sent images in different order, so the answers are inconsistent: return an uncertain answer
            return 0.5, [a1,a2]
        else:#both answers are different, which is consistent, because images were in different order: return the answer for the original image order sent as input to this function
            return a1, [a1,a2]
    else:
        if a1==0.5 and a2==0.5:#one of the 2 answers is uncertain: return an uncertain answer
            return 0.5, [a1,a2]
        elif a2==0.5:
            return a1, [a1,a2]
        elif a1==0.5:
            #invert a2
            if a2==1:    
                return 2, [a1,a2]
            else:
                return 1, [a1,a2]
        elif a1==a2:#both answers are the same, but I sent images in different order, so the answers are inconsistent: return an uncertain answer
            return 0.5, [a1,a2]
        else:#both answers are different, which is consistent, because images were in different order: return the answer for the original image order sent as input to this function
            return a1, [a1,a2]


def Prompt3MessagesSepFiguresLMDeployDualConfirmation(clean, y1, y2, 
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=DescriptionsED,
                            text_y1=ZeroShotInstructions, 
                            text_y2=ZeroShotInstructions,
                            text_compare=TextCompareAdd,
                            text_summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone',solid_overlay=False,
                            conservative=True):
    #sends 2 prompts, each one with one image order, checks if the answer is the same, if not, returns 0.5
    
    organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
    if organ=='liver':
        organRegion+=LiverDescriptionLocation
    if organ=='gall_bladder':
        organRegion+=GallbladderDescriptionLocation
    text_compare=text_compare % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    if 'skeleton' in window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1, [1,2]
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2, [2,1]
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
            else:
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1, [1,2]
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2, [2,1]
    
    if AnswerNo:
        a1=RedArea(y1)
        a2=RedArea(y2)
        print('Annotation should be zero')
        if a1==0 and a2==0:
            return 0.5, [1,1]
        elif a1==0:
            return 1, [1,2]
        elif a2==0:
            return 2, [2,1]
        else:
            return 0.5, [-1,-1]
    #else:
    #    a1=RedArea(y1)
    #    a2=RedArea(y2)
    #    if a1==0 and a2==0:
    #        print('Both annotations are empty, but they should not be')
    #        return 0.5, [1,1]
    #    elif a1==0:
    #        print('y1 is empty, but it should not be')
    #        return 2.0, [2,1]
    #    elif a2==0:
    #        print('y2 is empty, but it should not be')
    #        return 1.0, [1,2]
    
    
    text_y1 = text_y1 % {'organ': organ.replace('_',' '), 'number': 1} 
    if isinstance(organ_descriptions[organ], list):
        text_y1 += organ_descriptions[organ][0]
    else:
        text_y1 += organ_descriptions[organ]

    text_y2 = text_y2 % {'organ': organ.replace('_',' '), 'number': 2} 
    if isinstance(organ_descriptions[organ], list):
        text_y2 += organ_descriptions[organ][1]
    else:
        text_y2 += organ_descriptions[organ]

    if save_memory:
        conversation=[]

    #Analyze image 1
    imgs=[[y1],[y2]]
    conversation = [conversation,conversation]
    text = [text_y1,text_y1]

    conversation, answer = SendMessageLmdeploy(imgs,text=text, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay,
                                                batch=2)
    
    #Analyze image 2 and compare
    imgs=[[y2],[y1]]
    text = [text_y2+'\n'+text_compare, text_y2+'\n'+text_compare]
    conversation, answer = SendMessageLmdeploy(imgs,text=text, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay,
                                                batch=2)
    

    text = [text_summarize+answer[0],text_summarize+answer[1]]
    conversation, answer = SendMessageLmdeploy([[],[]], text=text, conver=[[],[]],
                                               base_url=base_url, size=size,
                                                batch=2)
    


    return CompareAnswers(answer,conservative)
    



Compare2ImagesGOOD='Check out these 2 images. In which one does the red shape come further up, image 1 or image 2?'
#90% accuracy!!!!!!!!!
Compare2ImagesBad=('Check out these 2 images. The best image is the one where the red shape comes further up. In which one does the red shape come further up, image 1 or image 2, or are they equal? If they are equal, answer "equal", if not, answer "the best is image 1" or "the best is image 2".')
                #'If one of the images has a red shape that comes further up, you must say that it is better and stop here. If both images have the red shape touching the image top, move to the next step.\n'
                #'Step 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red overlay reach the lumbar spine region? \n'
                #'If the red overlay reaches the lumbar spine region in just one of the images, you must say that it is better and stop here. If both images reach the lumbar spine region, move to the next step.\n'
                #'Step 3: In which image is the red overlay continuos and tubular?\n'
                #'If just one of the images has a continuos and tubular red overlay, you must say that it is better. If both images have a continuos and tubular red overlay, move to the next step.\n'
                #'Step 4: In which image is the red overlay more centered and parallel to the spine?\n'
                #'If just one of the images has a more centered and parallel red overlay, you must say that it is better.')
#Accuracy 60%
#do this step by step, in each step you sen just the 2 images (1 standalone prompt per question)




Compare2ImagesStomachV0="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays."
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2. 
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, or a sac with a downwards curvature.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay should be a single connected structure. If it has multiple structures or small disconnected parts, the overlay has a big error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#80%, 0 rejections
#try accepting unknown



Compare2ImagesStomachV2="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays."
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1, the one in Image 2, or neither. 
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple structures is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#85%, 1 rejection

Compare2ImagesStomachV3="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays."
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2. 
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass (two rounded bulbs separated by a narrow waist).
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#dual confirmation: 92%, 2 rejections, 1 mistake




Compare2ImagesStomachV5="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2.
'Say that there is no better overlay only if: you are totally sure that both are equally bad / you are totally sure they are equally good / both overlays have very big mistakes / you absolutelly cannot tell which one is better.'
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#Dual confirmation: 1 error, 3 refusals



Compare2ImagesStomachConservativeV0="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2.
Say that there is no better overlay only if: you are sure that both are equally bad / you are sure they are equally good / both overlays have very obvious mistakes / you truly cannot tell which one is better.'
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#Dual confirmation: accuracy 100%, 4 rejections


Compare2ImagesSpleenV0="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural curve of the spleen.
b) Shape 2: The spleen red overlay should not have irregular or random shapes. It should not include internal gaps or sharp, angular points.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney."""
# 0 errors, 2 rejections, th 0.7 dice-GPT round 0

Compare2ImagesSpleenV1="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The spleen red overlay should resemble an oval or crescent, with smooth and continuous curves. An irregular or jagged outline is likely incorrect, even if the overlay appears smaller.
b) Completeness: The spleen red overlay must cover the entire spleen. Any missing portions, especially in the upper or lateral aspects, indicate an incomplete and thus incorrect overlay.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures or disconnected sections are major errors.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and adjacent to the stomach and left kidney, just below the diaphragm. 
e) Size: The spleen red overlay may vary in size, depending on anatomical factors such as splenic enlargement. A larger spleen that still maintains the correct shape and location should not be considered incorrect solely due to size. Small overlays with irregular or jagged shapes are more likely to be incorrect.
f) Alignment: The overlay should follow the curvature of the ribs without extending too far toward the midline (spine) or downwards into the pelvic area.
g) Disease-related changes: If the spleen appears enlarged in one of the images, consider that this could be a result of an underlying medical condition. Enlarged spleens that maintain correct shape, smoothness, and continuity are still considered correct."""
#gpt round 1 - 1 mistake, 4 rejections th 8


Compare2ImagesSpleenV2="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It has only smooth and continuous curves, without recesses, sharp points or holes.
b) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
c) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh."""
# my turn: 1 mistake and one rejection on th 0.7

Compare2ImagesSpleenV3="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural curve of the spleen.
b) Shape 2: The spleen red overlay should have smooth curves only. It should not have irregular or random shapes, gaps, recesses or sharp points.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney."""
# more focus on smoothness- 5 rejections,

Compare2ImagesSpleenV4="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural curve of the spleen.
b) Smoothness: The spleen red overlay should have smooth curves only. It must NOT have gaps, holes or recesses.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney."""
#Acc: 4 /( 12 - 6 )

Compare2ImagesSpleenV5="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural and SMOOTH curve of the spleen.
b) Shape 2: The spleen red overlay should not have irregular or random shapes. It should not include internal gaps or sharp, angular points.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney."""
#2 rejections, 2 errors



Compare2ImagesGallbladderV0="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The gallbladder red overlay should resemble a small pear-shaped sac or oval, with a slight taper towards one end.
b) Size: The gallbladder red overlay should be relatively small, not extending beyond the expected size of a few centimeters in length.
c) Unity: The gallbladder red overlay must be a single connected structure. Multiple disconnected structures are a significant error.
d) Location: The gallbladder red overlay should be located in the upper right quadrant of the abdomen (left side of the image, which is oriented like an AP X-ray), beneath the liver, near the right edge of the ribcage. It should not extend too far towards the midline or too low towards the pelvis."""
#Errors:
# 2017- one gallbladder was too large, it chose the smaller overlay, which was wrong-BDMAP_V0002017; 
# 1406- model chose the very elongated shape, which was wrong and had a s curve;
# 1777 - chose a fragmented one
# 1889 - chose a small fragmented instead of a larger one
# 1645 - super difficult
# 2196 - super difficult, chose by size

#rejections: BDMAP_A0001660

Compare2ImagesGallbladderV1="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the gallbladder, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the gallbladder, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The gallbladder red overlay should resemble a pear or oval, and have a smooth shape.
b) Location: The gallbladder red overlay should be located in the upper right quadrant of the abdomen (left side of the figure, which is oriented like an AP X-ray). It should be near the bottom of the ribcage.
c) Unity: The gallbladder should be represented as a single red object. Multiple objects or disconnected parts are a big error."""



Compare2ImagesGallbladderV2="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the gallbladder, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the gallbladder, the one in Image 1 or in Image 2.
Say that there is no better overlay only if: you are sure that both are equally bad / you are sure they are equally good / both overlays have very obvious mistakes / you truly cannot tell which one is better.
Consider the following anatomical information:
a) Shape: The gallbladder red overlay should be pear-shaped or an elongated curved sack.
b) Shape 2: The gallbladder red overlay should be smooth. It should not have many random points.
c) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""

Compare2ImagesGallbladderV3="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the gallbladder, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the gallbladder, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The gallbladder red overlay should be pear-shaped or an elongated curved sack.
b) Shape 2: The surface of the gallbladder red overlay should be smooth. It should not have many sharp, angular points.
c) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""
#Accuracy:  0.8068181818181818
#Acc: 71 /( 110 - 22 )


Compare2ImagesGallbladderV4="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the gallbladder, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the gallbladder, the one in Image 1 or in Image 2. You may say neither only if you are sure that both are equally bad, equally good, or you are very unsure about which one is better.
Consider the following anatomical information:
a) Shape: The gallbladder red overlay should be pear-shaped or an elongated curved sack.
b) Shape 2: The gallbladder red overlay should be smooth. It should not have many random points.
c) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""
#77%
Compare2ImagesKidneysOld=("Check out these 2 images, and answer the following questions. I want you to conclude which image is better. The questions represent positive qualities that the images should satisfy. "
                   "a) Consider that a person usually has two kidneys. Does Image 1 show two distinct kidney overlays? Does Image 2 show two distinct kidney overlays? If both show a single kidney, the patient may truly have only one kidney. If only one images shows a single kidney, the image showing two kidneys should be better. \n"
                   "b) Shape: each kidney has a bean-shaped structure, with a convex lateral surface and a concave medial surface. Are the shapes of the 2 kidneys in image 1 correct? Are the shapes in image 2 better, worse or similar? \n"
                   "c) Completeness: does one image show more complete kidney overlays than the other? Is any overlay missing some part? \n"
                   "d) Location: the kidneys are located in the retroperitoneal space, on either side of the spine, at the level of the lower ribs. Are the locations of the 2 kidneys in image 1 correct? Are the kidney locations in image 2 better, worse or similar? \n")

                #'If one of the images has a red shape that comes further up, you must say that it is better and stop here. If both images have the red shape touching the image top, move to the next step.\n'
                #'Step 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red overlay reach the lumbar spine region? \n'
                #'If the red overlay reaches the lumbar spine region in just one of the images, you must say that it is better and stop here. If both images reach the lumbar spine region, move to the next step.\n'
                #'Step 3: In which image is the red overlay continuos and tubular?\n'
                #'If just one of the images has a continuos and tubular red overlay, you must say that it is better. If both images have a continuos and tubular red overlay, move to the next step.\n'
                #'Step 4: In which image is the red overlay more centered and parallel to the spine?\n'
                #'If just one of the images has a more centered and parallel red overlay, you must say that it is better.')
















Compare2ImagesAorta=('Check out these 2 images, and answer the following questions. I want you to conclude which image is better. Point 1 is the most important for determining this, then point 2, then point 3, and so on. '
                'Point 1: The best image is the one where the red shape comes further up. In which one does the red shape come further up, image 1 or image 2? '
                'Point 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red shape comes down to the lumbar spine height? '
                'Point 3: In which image is the red shape continuos and tubular? ')
#100% accuracy, halleluia!!!!!!!!!!!

Compare2ImagesFullAorta=('Check out these 2 images, and answer the following questions. I want you to conclude which image is better. I want you to conclude which image is better based on the answers to these points. '
                'Point 1: In which one does the red shape reach the thoracic region (high ribs), showing a curve in this area, image 1 or image 2?'
                'Point 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red shape comes down to the lumbar spine height?'
                'Point 3: In which image is the red shape continuos and tubular?')

Compare2ImagesPostcava=(
                'Check out these 2 images, and answer the following questions. I want you to conclude which image is better. The points represent positive qualities that the images should satisfy. '
                'Point 1: In which one does the red shape reach the thoracic region (high ribs), image 1 or image 2?'
                'Point 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red shape comes down to the lumbar spine height?'
                'Point 3: In which image is the red shape continuos and tubular?')

Compare2ImagesKidneys=("I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays."
                       " A red shape (overlay) over the images demarks the kidneys, but they may not be accurate. The overlays in Image 1 and Image 2 are different. "
                       "Evaluate each image individually, carefully compare them and conclude which overlay better represents the kidneys, the one in Image 1 or in Image 2. "
                       "Consider the following anatomical information: A person usually has two kidneys, check if the image display one, two or more red objects, this is a very important point. "
                      "Each kidney has a bean-shaped structure, with a slightly concave surface facing the spine, and a clearly convex surface facing outward. Check if the red objects resemble this shape and are complete. "
                      " The kidneys are located on either side of the spine, at the level of the lower ribs. Check if the red objects, if a pair, are on either side of the spine and at the level of the lower ribs. \n")

Compare2ImagesLiver=("I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays."
                       " A red shape (overlay) over the images demarks the liver, but they may not be accurate. The overlays in Image 1 and Image 2 are different. "
                       "Evaluate each image individually, carefully compare them and conclude which overlay better represents the liver, the one in Image 1 or in Image 2. "
                       "Consider the following anatomical information:"
                        "a) The liver is a large organ, with triangular or wedge-like shape."
                        "b) The liver is located in the upper right quadrant of the abdomen (left of the figure, like an AP X-ray), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. The liver is not near the pelvis."
                        "c) The liver position is primarily under the rib cage. The overlay must show red in the ribs region.")

Compare2ImagesStomachConservative="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2. You may say neither only if you are sure that both are equally bad, equally good, or you are very unsure about which one is better.
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#Dual confirmation: accuracy 100%, 4 rejections

Compare2ImagesStomach="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2. 
Consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#Dual confirmation: 93%, 1 rejection, 1 mistake --------Best until now

Compare2ImagesPancreas="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the pancreas, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the pancreas, the one in Image 1 or in Image 2. 
Consider the following anatomical information:
a) Shape: The pancreas is an elongated organ with a tadpole-like shape. The pancreas head is its thickest part and points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray. The other side of the pancreas is thin.
b) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. The organ is mostly horizontal, but may be slightly curved and its head usually sits lower than its tail.
c) Smoothness: The pancreas is a single smooth shape and it does not have very sharp edges."""

Compare2ImagesSpleen="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the spleen, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the spleen, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural curve of the spleen, with no small recesses.
b) Shape 2: The spleen red overlay should not have irregular or random shapes. It should not include internal gaps or sharp, angular points.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney."""
#2 rejections, 0 errors

Compare2ImagesGallbladder="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the gallbladder, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the gallbladder, the one in Image 1 or in Image 2.
Consider the following anatomical information:
a) Shape: The gallbladder red overlay should be pear-shaped or an elongated curved sack.
b) Shape 2: The gallbladder red overlay should be smooth. It should not have many random points.
c) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""
#Accuracy:  0.8433734939759037
#Acc: 70 /( 110 - 27 )
#Out of 13 mistakes, 2 are probable annotation errors: V0001777 and A0000686
#5 are certain label errors: A0001441, A0001802, A0001880, A0002308, V0000548
#This makes our acc at least: 75/(110-27)=75/83=0.9036144578313253

Compare2Images={
    'descending aorta':Compare2ImagesAorta,#better here
    'aorta':Compare2ImagesFullAorta,#better here
    'liver':Compare2ImagesLiver,
    'postcava':Compare2ImagesPostcava,#better here
    'kidneys':Compare2ImagesKidneys,#worst than putting one image per prompt and sending more prompts
    'stomach':Compare2ImagesStomach,#much better than putting one image per prompt and sending more prompts
    'pancreas':Compare2ImagesPancreas,
    'spleen':Compare2ImagesSpleen,
    'gall_bladder':Compare2ImagesGallbladder
}

Compare2ImagesSimple={
    'descending aorta':'These two images are frontal projecttions of the same CT scan. They look like and are oriented like X-rays. In which of these 2 images the red overlay better represents the aorta shape and location?',#better here
    'aorta':'These two images are frontal projecttions of the same CT scan. They look like and are oriented like X-rays. In which of these 2 images the red overlay better represents the aorta shape and location?',#better here
    'liver':Compare2ImagesLiver[:Compare2ImagesLiver.rfind('Consider the following')],
    'postcava':'These two images are frontal projecttions of the same CT scan. They look like and are oriented like X-rays. In which of these 2 images the red overlay better represents the postcava shape and location?',#better here
    'kidneys':Compare2ImagesKidneys[:Compare2ImagesKidneys.rfind('Consider the following')],#worst than putting one image per prompt and sending more prompts
    'stomach':Compare2ImagesStomach[:Compare2ImagesStomach.rfind('Consider the following')],#much better than putting one image per prompt and sending more prompts
    'pancreas':Compare2ImagesPancreas[:Compare2ImagesPancreas.rfind('Consider the following')],
    'spleen':Compare2ImagesSpleen[:Compare2ImagesSpleen.rfind('Consider the following')],
    'gall_bladder':Compare2ImagesGallbladder[:Compare2ImagesGallbladder.rfind('Consider the following')]
}

Compare2ImagesStomachShapeless="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the stomach, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the stomach, the one in Image 1 or in Image 2. 
Consider the following anatomical information, but do NOT consider the shape of the stomach overlay:
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""


Compare2ImagesPancreasShapeless="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the pancreas, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them and conclude which overlay better represents the pancreas, the one in Image 1 or in Image 2. 
Consider the following anatomical information, but do NOT consider the shape of the pancreas overlay:
a) Location: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
c) Unity: The pancreas red overlay must be a single connected structure. Showing multiple strucutres is a major error."""

Compare2ImagesGallbladderShapeless="""I am sending you 2 images, Image 1 and Image 2. Both images are frontal projections of the same CT scan. They are not CT slices, they have transparency, showing through the entire body. They look like AP X-rays.
A red shape (overlay) over the images demarks the gallbladder, but they may not be accurate. The overlays in Image 1 and Image 2 are different. 
Evaluate each image individually, carefully compare them, and conclude which overlay better represents the gallbladder, the one in Image 1 or in Image 2.
Consider the following anatomical information, but do NOT consider the shape of the gallbladder overlay:
a) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage."""

Compare2ImagesShapeless={
    'descending aorta':Compare2ImagesAorta,#better here
    'aorta':Compare2ImagesFullAorta,#better here
    'liver':Compare2ImagesLiver,
    'postcava':Compare2ImagesPostcava,#better here
    'kidneys':Compare2ImagesKidneys,#worst than putting one image per prompt and sending more prompts
    'stomach':Compare2ImagesStomachShapeless,
    'pancreas':Compare2ImagesPancreasShapeless,
    'spleen':Compare2ImagesSpleen,
    'gall_bladder':Compare2ImagesGallbladderShapeless
}


# In context learning comparisons

Compare2ImagesAortaInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, answer the following questions. The points represent positive qualities that the images should satisfy.
Point 1: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red shape comes down close to the lumbar spine height?
Point 2: In which image is the red shape continuos and tubular?
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. Notice that examples may have diverse field of views, some may show the aorta in its entirety (with a curve on top), while others may show only a portion of it, as is probably the case for the last 2 images. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesFullAortaInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, answer the following questions. The points represent positive qualities that the images should satisfy.
Point 1: In which one does the red shape reach the thoracic region (high ribs), showing a curve in this area, image X or image Y?
Point 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red shape comes down close to the lumbar spine height?
Point 3: In which image is the red shape continuos and tubular?
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. Notice that examples may have diverse field of views, some may show the aorta in its entirety (as is probably the case of the 2 images you must evaluate), while others may show only a portion of it.
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesPostcavaInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, answer the following questions. The points represent positive qualities that the images should satisfy.
Point 1: In which one does the red shape reach the thoracic region (high ribs), image X or image Y?
Point 2: Consider the bones in the image, both images display the same bones. Is the lumbar spine visible? If it is, in which image does the red shape comes down to the lumbar spine height?
Point 3: In which image is the red shape continuos and tubular?
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay.
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesKidneysInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information:
A person usually has two kidneys, check if the image display one, two or more red objects, this is a very important point.
Each kidney has a bean-shaped structure, with a slightly concave surface facing the spine, and a clearly convex surface facing outward. Check if the red objects resemble this shape and are complete.
The kidneys are located on either side of the spine, at the level of the lower ribs. Check if the red objects, if a pair, are on either side of the spine and at the level of the lower ribs.
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay.
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesLiverInContext = """The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information:
a) The liver is a large organ, with a triangular or wedge-like shape.
b) The liver is located in the upper right quadrant of the abdomen (left of the figure, like an AP X-ray), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. The liver is not near the pelvis.
c) The liver position is primarily under the rib cage. The overlay must show red in the ribs region.
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesStomachInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, a sac with a downwards curvature, or a hourglass.
b) Shape 2: The stomach red overlay should not be a random shape. It should not have many random points, nor internal gaps.
c) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs.
"As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay.
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesPancreasInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information:
a) Shape: The pancreas is an elongated organ with a tadpole-like shape. The pancreas head is its thickest part and points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray. The other side of the pancreas is thin.
b) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. The organ is mostly horizontal, but may be slightly curved and its head usually sits lower than its tail.
c) Smoothness: The pancreas is a single smooth shape and it does not have very sharp edges.
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesSpleenInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information:
a) Shape: The shape of the spleen red overlay should resemble an oval or crescent. It should follow the natural curve of the spleen, with no small recesses.
b) Shape 2: The spleen red overlay should not have irregular or random shapes. It should not include internal gaps or sharp, angular points.
c) Unity: The spleen red overlay must be a single, continuous structure. Multiple structures are a significant error.
d) Location: The spleen red overlay should be located in the upper left quadrant of the abdomen (right side of the image, which is oriented like an AP X-ray), slightly under the ribs and diaphragh, and adjacent to the stomach and left kidney.
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesGallbladderInContext="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information:
a) Shape: The gallbladder red overlay should be pear-shaped or an elongated curved sack.
b) Shape 2: The gallbladder red overlay should be smooth. It should not have many random points.
c) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
d) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage.
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesInContext={
    'descending aorta':Compare2ImagesAortaInContext,#better here
    'aorta':Compare2ImagesFullAortaInContext,#better here
    'liver':Compare2ImagesLiverInContext,
    'postcava':Compare2ImagesPostcavaInContext,#better here
    'kidneys':Compare2ImagesKidneysInContext,#worst than putting one image per prompt and sending more prompts
    'stomach':Compare2ImagesStomachInContext,#much better than putting one image per prompt and sending more prompts
    'pancreas':Compare2ImagesPancreasInContext,
    'spleen':Compare2ImagesSpleenInContext,
    'gall_bladder':Compare2ImagesGallbladderInContext
}



Compare2ImagesStomachInContextShapeless="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information, but do NOT consider the shape of the stomach overlay:
a) Unity: The stomach red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs.
"As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay.
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesPancreasInContextShapeless="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information, but do NOT consider the shape of the pancreas overlay:
a) Unity: The pancreas red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. 
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesGallbladderInContextShapeless="""The images I am sending are frontal projections of CT scans. They are not CT slices; rather, they have transparency, showing through the entire body. They are oriented like AP X-rays, where the right image side is the left side of the human body, and the left image side is the right side of the human body.
Red shapes (overlays) on the images demarcate the %(organ)s, but the accuracy of these overlays is uncertain. Your task is to compare the last two images I am sending, which we will call image X (penultimate image) and image Y (last image). These two images represent the same CT scan but have different %(organ)s overlays.
Evaluate each image individually, carefully compare the %(organ)s overlays, and determine which overlay better represents the %(organ)s: the one in image X or the one in image Y.
When comparing the images, consider the following anatomical information, but do NOT consider the shape of the gallbladder overlay:
a) Unity: The gallbladder red overlay must be a single connected structure. Showing multiple strucutres is a major error.
b) Location: The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray). It sits near the lower edge of the liver and the rib cage.
As examples to help you, I am sending you %(n)s other pairs of images. Each example pair contains one good overlay and one bad %(organ)s overlay. 
Again, you must compare Images X and Y, the last 2 I am sending (pair %(o)s)."""

Compare2ImagesInContextShapeless={
    'descending aorta':Compare2ImagesAortaInContext,#better here
    'aorta':Compare2ImagesFullAortaInContext,#better here
    'liver':Compare2ImagesLiverInContext,
    'postcava':Compare2ImagesPostcavaInContext,#better here
    'kidneys':Compare2ImagesKidneysInContext,#worst than putting one image per prompt and sending more prompts
    'stomach':Compare2ImagesStomachInContextShapeless,
    'pancreas':Compare2ImagesPancreasInContextShapeless,
    'spleen':Compare2ImagesSpleenInContext,
    'gall_bladder':Compare2ImagesGallbladderInContextShapeless
}





def Prompt2MessagesSepFiguresLMDeploy(clean, y1, y2, 
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=None,
                            text_compare=Compare2Images,
                            text_summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone',solid_overlay=False):
    
    organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    if organ=='liver':
        organRegion+=LiverDescriptionLocation
    if organ=='gall_bladder':
        organRegion+=GallbladderDescriptionLocation

    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    if 'skeleton' in window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
                text_compare=Compare2ImagesFullAorta
            else:
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
    
    if AnswerNo:
        a1=RedArea(y1)
        a2=RedArea(y2)
        print('Annotation should be zero')
        if a1==0 and a2==0:
            return 0.5
        elif a1==0:
            return 1
        elif a2==0:
            return 2
        else:
            return -1
    #else:
    #    a1=RedArea(y1)
    #    a2=RedArea(y2)
    #    if a1==0 and a2==0:
    #        print('Both annotations are empty, but they should not be')
    #        return 0.5
    #    elif a1==0:
    #        print('y1 is empty, but it should not be')
    #        return 2.0
    #    elif a2==0:
    #        print('y2 is empty, but it should not be')
    #        return 1.0
    
    
    if isinstance(text_compare, dict):
        text_compare=text_compare[organ]
    text_compare=text_compare % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}
    
    if save_memory:
        conversation=[]

    #Analyze image 1
    imgs=[y1,y2]
    conversation, answer = SendMessageLmdeploy(imgs,text=text_compare, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay)
    
    if 'image' not in answer.lower() and 'overlay' not in answer.lower():
        return 0.5
    
    conversation, answer = SendMessageLmdeploy([], text=text_summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5


def Prompt2MessagesSepFiguresLMDeployDualConfirmation(clean, y1, y2, 
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=None,
                            text_compare=Compare2Images,
                            text_summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone',solid_overlay=False,
                            conservative=False):

    print("=== Start Prompt2MessagesSepFiguresLMDeployDualConfirmation ===")
    print("Organ:", organ, "| Window:", window)
    print("Clean image:", clean)
    print("Overlay images: y1 =", y1, ", y2 =", y2)

    organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    if organ=='liver':
        organRegion+=LiverDescriptionLocation
    if organ=='gall_bladder':
        organRegion+=GallbladderDescriptionLocation

    print("\n[Step 1] Sending initial organRegion prompt...")
    print("Prompt text:", organRegion[:200], "..." if len(organRegion)>200 else "")
    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                               base_url=base_url, size=size)
    print("[Step 1] Raw answer:", answer)

    q='q2'
    if 'skeleton' in window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    print("AnswerNo detected:", AnswerNo, organ)

    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                print("Aorta special case → descending aorta")
                if red_on_top(y1) and not red_on_top(y2):
                    print('Red detected only on y1 → returning 1')
                    return 1, [1,2]
                elif red_on_top(y2) and not red_on_top(y1):
                    print('Red detected only on y2 → returning 2')
                    return 2, [2,1]
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
                text_compare=Compare2ImagesFullAorta
                print("Aorta special case → aortic arch present, using full aorta compare text")
            else:
                organ='descending aorta'
                print("Aorta special case → descending aorta")
                if red_on_top(y1) and not red_on_top(y2):
                    print('Red detected only on y1 → returning 1')
                    return 1, [1,2]
                elif red_on_top(y2) and not red_on_top(y1):
                    print('Red detected only on y2 → returning 2')
                    return 2, [2,1]

    if AnswerNo:
        a1=RedArea(y1)
        a2=RedArea(y2)
        print("\n[Step 2] Annotation should be zero check")
        print("RedArea(y1) =", a1, "| RedArea(y2) =", a2)
        if a1==0 and a2==0:
            return 0.5, [1,1]
        elif a1==0:
            return 1, [1,2]
        elif a2==0:
            return 2, [2,1]
        else:
            return 0.5, [-1,-1]

    if isinstance(text_compare, dict):
        text_compare=text_compare[organ]
    text_compare=text_compare % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if save_memory:
        conversation=[[],[]]
    else:
        conversation=[conversation,conversation]
    imgs=[[y1,y2],[y2,y1]]
    text=[text_compare,text_compare]

    print("\n[Step 2] Sending comparison prompt...")
    print("Imgs[0]:", imgs[0])
    print("Imgs[1]:", imgs[1])
    print("Prompt[0] preview:\n", text[0][:300], "..." if len(text[0])>300 else "")
    conversation, answer = SendMessageLmdeploy(imgs,text=text, conver=conversation,
                                               base_url=base_url, size=size, solid_overlay=solid_overlay,
                                               batch=2)
    print("[Step 2] Raw answer batch:", answer)

    imgs=[[],[]]
    text=[text_summarize+answer[0],text_summarize+answer[1]]
    conver=[[],[]]

    print("\n[Step 3] Sending summarization prompt...")
    print("Summarize prompt[0]:", text[0][:200], "..." if len(text[0])>200 else "")
    conversation, answer = SendMessageLmdeploy(imgs, text=text, conver=conver,
                                               base_url=base_url, size=size,
                                               batch=2)
    print("[Step 3] Raw summarization answer:", answer)

    final_answer = CompareAnswers(answer,conservative)
    print("=== Final parsed answer:", final_answer, "===")
    return final_answer


    
def Prompt2MessagesSepFiguresLMDeployDualConfirmationInContext(clean, y1, y2, 
                            good_examples, bad_examples,
                            good_examples_conf, bad_examples_conf,
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=None,
                            text_compare=Compare2ImagesInContext,
                            text_summarize=CompareSummarizeInContext, organ='liver',
                            save_memory=False, window='bone',solid_overlay=False,
                            conservative=False):
    
    organRegion=text_region % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder')}

    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    if organ=='liver':
        organRegion+=LiverDescriptionLocation
    if organ=='gall_bladder':
        organRegion+=GallbladderDescriptionLocation

    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    if 'skeleton' in window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1, [1,2]
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2, [2,1]
                
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
            else:
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1, [1,2]
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2, [2,1]
    
    if AnswerNo:
        a1=RedArea(y1)
        a2=RedArea(y2)
        print('Annotation should be zero')
        if a1==0 and a2==0:
            return 0.5, [1,1]
        elif a1==0:
            return 1, [1,2]
        elif a2==0:
            return 2, [2,1]
        else:
            return 0.5, [-1,-1]

    if isinstance(text_compare, dict):
        text_compare=text_compare[organ]
    text_compare=text_compare % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder'),'n':len(good_examples),'o':len(good_examples)+1}

    if save_memory:
        conversation=[[],[]]
    else:
        conversation=[conversation,conversation]
    imgs=[[],[]]
    text=[text_compare+'\n',text_compare+'\n']
    
    i=0
    for good_ex, bad_ex, good_ex_conf, bad_ex_conf in zip(good_examples, bad_examples, good_examples_conf, bad_examples_conf):
        i+=1
        best_ex=random.randint(1,2)
        best_ex_conf=random.randint(1,2)
        if best_ex==1:
            imgs[0].append(good_ex)
            imgs[0].append(bad_ex)
            text[0]+="Pair "+str(i)+": Image X is better\n"
        else:
            imgs[0].append(bad_ex)
            imgs[0].append(good_ex)
            text[0]+="Pair "+str(i)+": Image Y is better\n"
        if best_ex_conf==1:
            imgs[1].append(good_ex_conf)
            imgs[1].append(bad_ex_conf)
            text[1]+="Pair "+str(i)+": Image X is better\n"
        else:
            imgs[1].append(bad_ex_conf)
            imgs[1].append(good_ex_conf)
            text[1]+="Pair "+str(i)+": Image Y is better\n"
        
    imgs[0].append(y1)
    imgs[0].append(y2)
    imgs[1].append(y2)
    imgs[1].append(y1)
    text[0]+="Pair "+str(i+1)+": \n"
    text[1]+="Pair "+str(i+1)+": \n"

    conversation, answer = SendMessageLmdeploy(imgs,text=text, conver=conversation,
                                                base_url=base_url, size=size, solid_overlay=solid_overlay,
                                                batch=2)
    
    imgs=[[],[]]
    text_summarize=text_summarize % {'organ': organ.replace('_',' ').replace('gall bladder','gallbladder'),'n':len(good_examples),'o':len(good_examples)+1}
    text=[text_summarize+answer[0],text_summarize+answer[1]]
    conver=[[],[]]
    conversation, answer = SendMessageLmdeploy(imgs, text=text, conver=conver,
                                               base_url=base_url, size=size,
                                               batch=2)

    return CompareAnswers(answer,conservative,in_context=True)

def Prompt4MessagesSepFiguresLMDeploy(clean, y1, y2, 
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=DescriptionsED,
                            text_y1=ZeroShotInstructions, 
                            text_y2=ZeroShotInstructions,
                            text_compare=TextCompare,
                            text_summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone'):
    
    organRegion=text_region % {'organ': organ.replace('_',' ')}
    text_compare=text_compare % {'organ': organ.replace('_',' ')}
    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    if 'skeleton' in window:
        q='q4'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
            else:
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
    
    if AnswerNo:
        a1=RedArea(y1)
        a2=RedArea(y2)
        print('Annotation should be zero')
        if a1==0 and a2==0:
            return 0.5
        elif a1==0:
            return 1
        elif a2==0:
            return 2
        else:
            return -1
    #else:
    #    a1=RedArea(y1)
    #    a2=RedArea(y2)
    #    if a1==0 and a2==0:
    #        print('Both annotations are empty, but they should not be')
    #        return 0.5
    #    elif a1==0:
    #        print('y1 is empty, but it should not be')
    #        return 2.0
    #    elif a2==0:
    #        print('y2 is empty, but it should not be')
    #        return 1.0
    
    
    text_y1 = text_y1 % {'organ': organ.replace('_',' '), 'number': 1} 
    text_y1 += organ_descriptions[organ]

    text_y2 = text_y2 % {'organ': organ.replace('_',' '), 'number': 2} 
    text_y2 += organ_descriptions[organ]

    if save_memory:
        conversation=[]

    #Analyze image 1
    imgs=[y1]
    conversation, answer = SendMessageLmdeploy(imgs,text=text_y1, conver=conversation,
                                                base_url=base_url, size=size)
    
    #Analyze image 2
    imgs=[y2]
    conversation, answer = SendMessageLmdeploy(imgs,text=text_y2, conver=conversation,
                                                base_url=base_url, size=size)

    #compare
    imgs=[]
    conversation, answer = SendMessageLmdeploy(imgs,text=text_compare, conver=conversation,
                                                base_url=base_url, size=size)
    
    if 'image' not in answer.lower() and 'overlay' not in answer.lower():
        return 0.5
    
    conversation, answer = SendMessageLmdeploy([], text=text_summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5


def Prompt3MessagesLMDeploy(img1, img2, img3, 
                            base_url='http://0.0.0.0:8000/v1', size=512,
                            text1=BodyRegionText, 
                            textOrganPresent=ComparisonTextContinued, 
                            textOrganNotPresent=NoOrganSimple, 
                            summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone'):
    
    organRegion=text1 % {'organ': organ.replace('_',' ')}
    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    if size>224:
        conversation, answer = SendMessageLmdeploy([img1], conver=[], text=organRegion,
                                                base_url=base_url, size=224)
    else:
        conversation, answer = SendMessageLmdeploy([img1], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if 'skeleton' in window:
            if ('no' in answer.lower()[answer.lower().rfind('q6'):answer.lower().rfind('q6')+7]):#no lungs
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
            else:
                organ='descending aorta'
                #check for red on top of the images
                if red_on_top(y1) and not red_on_top(y2):
                    print('There should be red on top of the images, only image 1 has it')
                    return 1
                elif red_on_top(y2) and not red_on_top(y1):
                    print('There should be red on top of the images, only image 2 has it')
                    return 2
    
    if AnswerNo:
        #text2 = NoOrganText % {'organ': organ.replace('_',' ')}
        conversation, answer = SendMessageLmdeploy([img3],text=textOrganNotPresent, conver=[],
                                               base_url=base_url, size=size)
        if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
            return 1
        elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
            return 2
        else:
            return 0.5
    else:   
        text2 = textOrganPresent % {'organ': organ.replace('_',' ')} 
        text2 += Descriptions[organ]

    if save_memory:
        conversation=[]
        imgs=[img2]
    else:
        imgs=[img1,img2]

    conversation, answer = SendMessageLmdeploy(imgs,text=text2, conver=conversation,
                                                base_url=base_url, size=size)
    
    if 'overlay' not in answer.lower():
        return 0.5

    conversation, answer = SendMessageLmdeploy([], text=summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5


def MultiTurnMultiImageComparisonLMDeploy(clean, y1, y2, compImg, 
                            base_url='http://0.0.0.0:8000/v1', size=448,
                            text1=BodyRegionText, 
                            textOrganPresent=ComparisonText, 
                            textOrganNotPresent=NoOrganSimple, 
                            summarize=CompareSummarize, organ='liver',
                            save_memory=False):
    '''
    Idea: send one image, check for errors. Send another one, check for errors. Send both, compare.
    '''
    
    
    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=text1 % {'organ': organ.replace('_',' ')},
                                                base_url=base_url, size=256)
    
    AnswerNo=('no' in answer.lower()[answer.lower().rfind('q2'):answer.lower().rfind('q2')+15])
    
    if AnswerNo:
        #text2 = NoOrganText % {'organ': organ.replace('_',' ')}
        conversation, answer = SendMessageLmdeploy([compImg],text=textOrganNotPresent, conver=[],
                                               base_url=base_url, size=size)
        if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
            return 1
        elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
            return 2
        else:
            return 0.5
    else:   
        comp = textOrganPresent % {'organ': organ.replace('_',' ')} 
        comp += Descriptions[organ]

    textED=ZeroShotInstructions%{'organ':organ.replace('_',' ')}
    textED+=DescriptionsED[organ]
    textED+='Justify your answer but answer concisely.'

    conversation1, answer1 = SendMessageLmdeploy([y1], text='This is Image 1. '+textED, conver=[],
                                               base_url=base_url, size=size//2)
    
    conversation2, answer2 = SendMessageLmdeploy([y2], text='This is Image 2. '+textED, conver=[],
                                               base_url=base_url, size=size//2)
    
    conversation, answer = SendMessageLmdeploy([y1,y2,compImg],text=comp, conver=conversation1+conversation2,
                                                base_url=base_url, size=[size//4,size//4,size])
    
    if 'overlay' not in answer.lower():
        return 0.5

    conversation, answer = SendMessageLmdeploy([], text=summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5
    

def SaveDices(pth, organ='liver',file_list=None,
            ):
        
        best=2
        dice_csv_path=os.path.join(pth,'DSC.csv')
        if file_list is not None:
            with open(file_list, 'r') as file:
                file_list = file.readlines()
            # Removing the newline characters at the end of each line (optional)
            file_list = [line.strip() for line in file_list]

        #print('File list:',file_list)
        

        for target in os.listdir(pth):
            if file_list is not None:
                #print ('Target:',target[:14] )
                if target[:14] not in file_list:
                    #print('Skipping:',target)
                    continue
            
            if 'ct_window_bone_axis_1' not in target:
                continue

            clean=os.path.join(pth,target)

            y1=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y1.png')
            y2=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y2.png')

            print('y1:',y1)
            print('y2:',y2)
            

            if organ not in clean[clean.rfind('ct'):]:
                y1=y1.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')
                y2=y2.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')

            dice=check_dice(y1,y2)
            #save dice to csv
            with open(dice_csv_path, 'a') as f:
                f.write(f'{target},{dice}\n')
            print('Dice:',dice)

def SimpleMultiImageComparisonLMDeploy(clean, y1, y2, compImg, 
                            base_url='http://0.0.0.0:8000/v1', size=448,
                            text1=BodyRegionText, 
                            textOrganPresent=ComparisonText2FigsContinued, 
                            textOrganNotPresent=NoOrganSimple, 
                            summarize=CompareSummarize, organ='liver',
                            save_memory=False):
    '''
    Idea: send one image, check for errors. Send another one, check for errors. Send both, compare.
    '''
    
    
    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=text1 % {'organ': organ.replace('_',' ')},
                                                base_url=base_url, size=256)
    
    AnswerNo=('no' in answer.lower()[answer.lower().rfind('q2'):answer.lower().rfind('q2')+15])
    
    if AnswerNo:
        #text2 = NoOrganText % {'organ': organ.replace('_',' ')}
        conversation, answer = SendMessageLmdeploy([compImg],text=textOrganNotPresent, conver=[],
                                               base_url=base_url, size=size)
        if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
            return 1
        elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
            return 2
        else:
            return 0.5
    else:   
        comp = textOrganPresent % {'organ': organ.replace('_',' ')} 
        comp += Descriptions[organ]

    textED=ZeroShotInstructions%{'organ':organ.replace('_',' ')}
    textED+=DescriptionsED[organ]
    textED+='Justify your answer but answer concisely.'

    conversation=CreateConversation([clean], 'This is Image 0, a frontal projection of a CT scan, it has some transparency and looks like an X-ray. ',
                                    conver=[],size=size//4)
    
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    
    conversation.append({
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Indeed, this is a frontal projection of a CT scan. It is not a CT slice, instead, it has transparency and lets you see through the entire human body, like an X-ray. ',
            }],
        })
    
    conversation=CreateConversation([y1], 'This is Image 1, it is the same frontal projection of a CT scan I shown in image 0, but now there is a red overlay, overlay 1, which tries to display the {organ}. ',
                                    conver=conversation,size=size//4)
    
    conversation.append({
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Indeed, this is the same frontal projection of a CT scan, but now there is a red overlay, overlay 1.',
            }],
        })
    
    conversation=CreateConversation([y2], 'This is Image 2, it is the same frontal projection of a CT scan I shown in Image 0 and Image 1, but now there is another red overlay, overlay 2, which is different from overlay 1, but also tries to display the {organ}. ',
                                    conver=conversation,size=size//4)
    
    conversation.append({
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Indeed, this is the same frontal projection of a CT scan, but now there is a red overlay, overlay 2. Overlay 2 is different from overlay 1.',
            }],
        })
    
    conversation, answer = SendMessageLmdeploy([y1,y2,compImg],text=comp, conver=conversation,
                                                base_url=base_url, size=[size//4,size//4,size])
    
    if 'overlay' not in answer.lower():
        return 0.5

    conversation, answer = SendMessageLmdeploy([], text=summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5
    

def SystematicComparison3MessagesLMDeploy4ImageSequence(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent='auto', 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize, organ='liver',
                    file_structure='original',
                    dice_check=False,pth1=None,pth2=None,save_memory=False):
        
        answers=[]
        outputs={}

        if textOrganPresent=='auto':
            textOrganPresent=ComparisonText2FigsContinued
            textOrganNotPresent=NoOrganSimple

        for target in os.listdir(pth):
            if '_ct_' not in target:
                continue
            clean=os.path.join(pth,target)
            twoImages=clean.replace('ct_window_bone','composite_image_2_figs')
            y1=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y1.png')
            y2=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y2.png')
            #print(target)
            #print('clean:',clean)
            #print('y1:',y1)
            #print('y2:',y2)
            #print('twoImages:',twoImages)


            print(target)

            if dice_check:
                #print('pth:',pth)
                pid=target[:target.rfind('_ct')]
                #print(os.listdir(pth1))
                #print('pid:',pid)
                dice=check_dice(y1,y2)
                #dice=check_dice_on_composite_2_figs(twoImages)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>0.9:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct compare slices (holes?).')
                    continue

            #consider that the correct answer is 2
            answer=SimpleMultiImageComparisonLMDeploy(
                            clean=clean,y1=y1,y2=y2,compImg=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ,save_memory=save_memory)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)






            
def SystematicComparison3MessagesLMDeploy(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent='auto', 
                    textOrganNotPresent='auto', 
                    summarize=CompareSummarize, organ='liver',
                    file_structure='original',
                    dice_check=False,pth1=None,pth2=None,
                    save_memory=False):
        
        answers=[]
        outputs={}

        if textOrganPresent=='auto':
            if organ[-1]!='s':
                textOrganPresent=ComparisonText
                textOrganNotPresent=NoOrganSimple

            else:
                #textOrganPresent=ComparisonText2Classes%{'organ_singular':organ[:-1],'organ':organ}
                #textOrganNotPresent=NoOrganSimple2Classes
                textOrganPresent=ComparisonText
                textOrganNotPresent=NoOrganSimple

        for target in os.listdir(pth):
            if file_structure=='original':
                if 'overlay_axis_1' not in target or 'BestIs' in target:
                    continue
                anno=os.path.join(pth,target)
                clean=anno.replace('overlay','ct')
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                if 'ct_window_bone_axis_1' not in target:
                    continue
                clean=os.path.join(pth,target)
                anno=clean.replace('ct_window_bone','composite_image')
                twoImages=clean.replace('ct_window_bone','composite_image_2_figs')
                print('anno:',anno)

            print(target)

            if dice_check:
                #print('pth:',pth)
                pid=target[:target.rfind('_ct')]
                #print(os.listdir(pth1))
                #print('pid:',pid)
                #dice=check_dice(os.path.join(pth1,pid,pid+'_overlay_window_bone_axis_1_'+organ+'.png'),
                #                os.path.join(pth2,pid,pid+'_overlay_window_bone_axis_1_'+organ+'.png'))
                dice=check_dice_on_composite_2_figs(twoImages)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>0.9:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct compare slices (holes?).')
                    continue

            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=anno,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ, save_memory=save_memory)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)

def SystematicComparison3MessagesLMDeploy2Figs(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent='auto', 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize2Figs, organ='liver',
                    file_structure='original',
                    dice_check=False,pth1=None,pth2=None,save_memory=False,
                    window='bone',shuffle=True):
        
        if window=='skeleton':
            text1=BodyRegionTextHighlightedSkeleton
        answers=[]
        outputs={}

        if textOrganPresent=='auto':
            if organ[-1]!='s':
                textOrganPresent=ComparisonText2Figs
                textOrganNotPresent=NoOrganSimple
            else:
                #textOrganPresent=ComparisonText2Classes2Figs%{'organ_singular':organ[:-1],'organ':organ}
                #textOrganNotPresent=NoOrganSimple2Classes
                textOrganPresent=ComparisonText2Figs
                textOrganNotPresent=NoOrganSimple

        

        for target in os.listdir(pth):
            if shuffle:
                best=random.randint(1,2)
            else:
                best=2
            if file_structure=='original':
                best=2
                if shuffle:
                    raise ValueError('Shuffle is not implemented for original file structure.')
                if 'overlay_axis_1' not in target or 'BestIs' in target:
                    continue
                anno=os.path.join(pth,target)
                clean=anno.replace('overlay','ct')
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                if 'ct_window_bone_axis_1' not in target:
                    continue
                clean=os.path.join(pth,target)
                if best==2:
                    twoImages=clean.replace('ct_window_bone','composite_image_2_figs')
                else:
                    twoImages=clean.replace('ct_window_bone','best1_composite_image_2_figs')
            
                

            print(target)
            #print('Best is:',best)
            print('Annotation:',twoImages) 

            if dice_check:
                #print('pth:',pth)
                pid=target[:target.rfind('_ct')]
                #print(os.listdir(pth1))
                #print('pid:',pid)
                dice=check_dice(clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y1.png'),
                                clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y2.png'))
                #dice=check_dice_on_composite_2_figs(twoImages)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>0.9:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct compare slices (holes?).')
                    continue
            

            if window=='skeleton':
                #clean=clean[:clean.rfind('ct_window_bone')]+'composite_ct_2_figs_axis_1_skeleton.png'
                #clean=clean.replace('ct_window_bone','composite')
                clean=clean[:clean.rfind('ct_window_bone')]+'highlighted_skeleton.png'
                #clean=clean.replace('ct_window_bone','ct_window_skeleton')
                print('anno:',anno)
                print('clean:',clean)

            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=twoImages,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ,save_memory=save_memory,
                            window=window)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay '+str(best))
            if answer==best:
                answers.append(1)
            else:
                answers.append(0)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)



def check_case_exists(csv_file, case_name):
    import pandas as pd
    df=pd.read_csv(csv_file,header=None, names=['case', 'answer', 'label', 'correct', 'organ', 'answer_1', 'answer_2'])
    print('csv file:',csv_file)
    return case_name in df['case'].values
    

def SystematicComparisonLMDeploySepFigures(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text_region=BodyRegionText, 
                            organ_descriptions=DescriptionsED,
                            text_y1=FindErrors, 
                            text_y2=FindErrors,
                            text_compare=TextCompareAdd,
                            text_summarize=CompareSummarize2Figs, organ='liver',
                            dice_check=False,pth1=None,pth2=None,save_memory=False,
                            window='skeleton',shuffle=True,best=None,
                            superpose=False,comparison_window='bone',
                            solid_overlay='auto',multi_image_prompt_2='auto',
                            text_multi_image_prompt_2=Compare2Images,
                            dice_th=0.5,dice_threshold_max=0.9,file_list=None,
                            dual_confirmation=False,conservative_dual=False,
                            csv_file=None,restart=True,
                            examples=0,dice_list=None,
                            shapeless=False,simple_prompt_ablation=False,
                            min_cases=100,max_cases=10000):

        if examples>0:
            text_compare=Compare2ImagesInContext
            text_summarize=CompareSummarizeInContext

        if shapeless:
            organ_descriptions=DescriptionsEDShapeless
            text_multi_image_prompt_2=Compare2ImagesShapeless
            text_y1=FindErrorsShapeless
            text_y2=FindErrorsShapeless
            if examples>0:
                text_compare=Compare2ImagesInContextShapeless

        if simple_prompt_ablation:
            tmp={}
            for k,v in organ_descriptions.items():
                tmp[k]=''
            organ_descriptions=tmp
            text_multi_image_prompt_2=Compare2ImagesSimple
            if examples>0:
                raise ValueError('Examples not implemented for simple prompt ablation.')

        

        if csv_file is not None:
            column_names = ['case', 'answer', 'label', 'correct', 'organ', 'answer_1', 'answer_2']
            if os.path.exists(csv_file) and restart:
                os.remove(csv_file)
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)
            elif not os.path.exists(csv_file):
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(column_names)


        if dice_list is not None:
            import pandas as pd
            print('Loading dice list:',dice_list)
            df=pd.read_csv(dice_list,header=None, names=["case", "dice"])

            #change dice threshold according to mean and std of dice in list
            #a=df['dice'].mean()-df['dice'].std()
            #order dice in ascending order and get the 100th value
            #b=df['dice'].sort_values().iloc[100]
            #dice_th=max(dice_th,min(a,b))
            dice_th=min(df['dice'].mean()-df['dice'].std(),dice_threshold_max)
            print('Dice threshold re-defined to:',dice_th)

            filtered_df = df[df["dice"] < dice_th]
            # Get the list of 'case' values where 'dice' is below the threshold
            cases_below_threshold = filtered_df["case"].tolist()
            cases_high_dice=df[df["dice"] >= 0.95]["case"].tolist()
            if len(cases_below_threshold)<min_cases:
                #if too few cases, get 100 lowest dice cases to compare
                dice_th=df['dice'].sort_values().iloc[min_cases]
                filtered_df = df[df["dice"] < dice_th]
                # Get the list of 'case' values where 'dice' is below the threshold
                cases_below_threshold = filtered_df["case"].tolist()
            if len( cases_below_threshold)>max_cases:
                #limit to 10% of dataset
                dice_th=df['dice'].sort_values().iloc[max_cases]
                filtered_df = df[df["dice"] < dice_th]
                # Get the list of 'case' values where 'dice' is below the threshold
                cases_below_threshold = filtered_df["case"].tolist()

            print('Number of cases to check:',len(cases_below_threshold))


        
        if multi_image_prompt_2=='auto':
            if organ in ['kidneys','liver','pancreas']:
                multi_image_prompt_2=False
            elif organ in ['aorta','postcava','spleen','stomach','gall_bladder']:
                multi_image_prompt_2=True
            else:
                raise ValueError('Organ not recognized for multi_image_prompt_2=auto')
            
        if solid_overlay=='auto':
            if organ in ['aorta','postcava']:
                solid_overlay=True
            else:
                solid_overlay=False
            
        if window=='skeleton':
            text_region=BodyRegionTextSkeleton
        if comparison_window=='skeleton':
            text_y1=FindErrorsSkeleton
            text_y2=FindErrorsSkeleton

        answers=[]
        labels=[]
        outputs={}

        if file_list is not None:
            with open(file_list, 'r') as file:
                file_list = file.readlines()
            # Removing the newline characters at the end of each line (optional)
            file_list = [line.strip() for line in file_list]

        #print('File list:',file_list)
        
        if dice_list is not None and dice_check:
            cases=cases_below_threshold
        else:
            cases=os.listdir(pth)

        for target in cases:

            if file_list is not None:
                #print ('Target:',target[:14] )
                if target[:14] not in file_list:
                    #print('Skipping:',target)
                    continue
            if shuffle:
                best=random.randint(1,2)
            
            if 'ct_window_bone_axis_1' not in target:
                continue
            clean=os.path.join(pth,target)

            if ((csv_file is not None) and (not restart)):
                if check_case_exists(csv_file, target):
                    print('Already exists, skipping case ',target)
                    continue
            
            if best==2:
                y1=clean.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')
                y2=clean.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')
            else:
                y1=clean.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')
                y2=clean.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')

            if organ not in clean[clean.rfind('ct'):]:
                y1=y1.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')
                y2=y2.replace('_y1.png','_'+organ+'_y1.png').replace('_y2.png','_'+organ+'_y2.png')

            if superpose:
                text_compare=TextCompareSuper
                y_super=superpose_images(y1,y2)
                from io import BytesIO
                fake_file = BytesIO()
                y_super.save(fake_file, format='PNG')
                fake_file.seek(0)


            print(target)
            print('Best is:',best)

            if dice_check:# and dice_list is None:
                #only check dice if no dice list is provided, otherwise we already filtered the cases before the loop
                dice=check_dice(y1,y2)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>dice_th:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct slices (holes?).')
                    continue
            
            if window=='skeleton':
                clean=clean.replace('ct_window_bone','ct_window_skeleton')
                print('clean:',clean)

            if dual_confirmation and superpose:
                raise ValueError('Dual confirmation is not implemented for superpose or multi_image_prompt_2.')
            

            print('dual_confirmation:',dual_confirmation)
            print('multi_image_prompt_2:',multi_image_prompt_2)
            print('examples:',examples)
            if examples>0:
                good_examples=random.sample(cases_high_dice,examples)
                #use y2 over all good examples
                good_examples=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')) for x in good_examples]
                bad_examples=random.sample([x for x in cases_below_threshold if x!=target],examples)
                #use y1 over all bad examples
                bad_examples=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')) for x in bad_examples]
                
                good_examples_conf=random.sample(cases_high_dice,examples)
                #use y2 over all good examples
                good_examples_conf=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')) for x in good_examples_conf]   
                bad_examples_conf=random.sample([x for x in cases_below_threshold if x!=target],examples)
                #use y1 over all bad examples
                bad_examples_conf=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')) for x in bad_examples_conf]



                #examples_sel=random.sample(cases_below_threshold,examples)
                #good_examples=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')) for x in examples_sel]
                #bad_examples=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')) for x in examples_sel]

                #good_examples_conf=random.sample(cases_high_dice,(examples-examples//2))+random.sample(cases_below_threshold,examples//2)
                #use y2 over all good examples
                #good_examples_conf=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')) for x in good_examples_conf]
                #bad_examples_conf=random.sample(cases_below_threshold,examples)
                #use y1 over all bad examples
                #bad_examples_conf=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')) for x in bad_examples_conf]

                #examples_conf=random.sample(cases_below_threshold,examples)
                #good_examples_conf=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y2.png')) for x in examples_conf]
                #bad_examples_conf=[os.path.join(pth,x.replace('ct_window_bone','overlay_window_'+comparison_window).replace('.png','_y1.png')) for x in examples_conf]



                if organ not in clean[clean.rfind('ct'):]:
                    good_examples=[x.replace('_y2.png','_'+organ+'_y2.png') for x in good_examples]
                    bad_examples=[x.replace('_y1.png','_'+organ+'_y1.png') for x in bad_examples]
                    good_examples_conf=[x.replace('_y2.png','_'+organ+'_y2.png') for x in good_examples_conf]
                    bad_examples_conf=[x.replace('_y1.png','_'+organ+'_y1.png') for x in bad_examples_conf]


                if dual_confirmation:
                    answer,answer_dual=Prompt2MessagesSepFiguresLMDeployDualConfirmationInContext(
                                    clean=clean,y1=y1,y2=y2,
                                    bad_examples=bad_examples,good_examples=good_examples,
                                    bad_examples_conf=bad_examples_conf,good_examples_conf=good_examples_conf,
                                    base_url=base_url,size=size,
                                    text_region=text_region, 
                                    organ_descriptions=organ_descriptions,
                                    text_compare=text_compare,
                                    text_summarize=text_summarize,
                                    organ=organ,save_memory=save_memory,
                                    window=window,solid_overlay=solid_overlay,
                                    conservative=conservative_dual)
                else:
                    raise ValueError('Examples are only implemented for dual confirmation and multiple images in one prompt.')

            else:
                if dual_confirmation:
                    if multi_image_prompt_2:
                        print('Using dual confirmation and sending images together.')
                        answer,answer_dual=Prompt2MessagesSepFiguresLMDeployDualConfirmation(
                                    clean=clean,y1=y1,y2=y2,
                                    base_url=base_url,size=size,
                                    text_region=text_region, 
                                    organ_descriptions=organ_descriptions,
                                    text_compare=text_multi_image_prompt_2,
                                    text_summarize=text_summarize,
                                    organ=organ,save_memory=save_memory,
                                    window=window,solid_overlay=solid_overlay,
                                    conservative=conservative_dual)
                    else:
                        print('Using dual confirmation and sending images separately.')
                        answer,answer_dual=Prompt3MessagesSepFiguresLMDeployDualConfirmation(
                                        clean=clean,y1=y1,y2=y2,
                                        base_url=base_url,size=size,
                                        text_region=text_region, 
                                        organ_descriptions=organ_descriptions,
                                        text_y1=text_y1, 
                                        text_y2=text_y2,
                                        text_compare=text_compare,
                                        text_summarize=text_summarize,
                                        organ=organ,save_memory=save_memory,
                                        window=window,solid_overlay=solid_overlay,
                                        conservative=conservative_dual)
                elif superpose:
                    answer=Prompt4MessagesSepFiguresLMDeploySuperposition(
                                clean=fake_file,y1=y1,y2=y2,y_super=fake_file,
                                base_url=base_url,size=size,
                                text_region=text_region, 
                                organ_descriptions=organ_descriptions,
                                text_y1=text_y1, 
                                text_y2=text_y2,
                                text_compare=text_compare,
                                text_summarize=text_summarize,
                                organ=organ,save_memory=save_memory,
                                window=window,solid_overlay=solid_overlay)
                elif multi_image_prompt_2:
                    answer=Prompt2MessagesSepFiguresLMDeploy(
                                    clean=clean,y1=y1,y2=y2,
                                    base_url=base_url,size=size,
                                    text_region=text_region, 
                                    organ_descriptions=organ_descriptions,
                                    text_compare=text_multi_image_prompt_2,
                                    text_summarize=text_summarize,
                                    organ=organ,save_memory=save_memory,
                                    window=window,solid_overlay=solid_overlay)

                else:
                    answer=Prompt3MessagesSepFiguresLMDeploy(
                                    clean=clean,y1=y1,y2=y2,
                                    base_url=base_url,size=size,
                                    text_region=text_region, 
                                    organ_descriptions=organ_descriptions,
                                    text_y1=text_y1, 
                                    text_y2=text_y2,
                                    text_compare=text_compare,
                                    text_summarize=text_summarize,
                                    organ=organ,save_memory=save_memory,
                                    window=window,solid_overlay=solid_overlay)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay '+str(best), 'Correct:',best==answer)
            if csv_file is not None:
                if dual_confirmation:
                    row = [target, answer, str(best), best==answer, organ, answer_dual[0], answer_dual[1]]
                else:
                    row = [target, answer, str(best), best==answer, organ, '-', '-']

                # Append the row to the CSV file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
            answers.append(answer)
            labels.append(best)
            if dual_confirmation:
                outputs[target]=[(best==answer),answer_dual]
            else:
                outputs[target]=(best==answer)

            if superpose:
                fake_file.close()
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        #calculate accuracy based on answers and labels
        answers=np.array(answers)
        labels=np.array(labels)

        acc=(answers==labels).sum()/(len(answers)-np.where(answers==0.5)[0].shape[0])
        print('Accuracy: ',acc)
        print('Acc:',(answers==labels).sum(),'/(',len(answers),'-',np.where(answers==0.5)[0].shape[0],')')
        print('answers:',answers)
        print('labels:',labels)
        print()

        for k,v in outputs.items():
            print(k,v)
organ_list=['adrenal_gland_left',
 'adrenal_gland_right',
 'aorta',
 'bladder',
 'celiac_trunk',
 'colon',
 'duodenum',
 'esophagus',
 'femur_left',
 'femur_right',
 'gall_bladder',
 'hepatic_vessel',
 'intestine',
 'kidney_left',
 'kidney_right',
 'liver',
 'lung_left',
 'lung_right',
 'pancreas',
 'portal_vein_and_splenic_vein',
 'postcava',
 'prostate',
 'rectum',
 'spleen',
 'stomach']

# Function to calculate Dice coefficient
def calculate_dice(mask1, mask2):
    intersection = torch.sum(mask1 & mask2)
    total = torch.sum(mask1) + torch.sum(mask2)
    dice = (2.0 * intersection) / (total+1e-6)
    return dice.item()

# Function to process the image
def check_dice_on_composite_2_figs(image_path):
    import cv2
    # 1- Load the image
    img = np.array(Image.open(image_path))
    
    # Convert image to grayscale to simplify background detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold to create a binary image (X-rays are darker, so we invert the threshold)
    _, binary_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect the X-ray areas
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and get the largest two (assuming these are the X-rays)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Extract bounding boxes for the two largest contours
    xray_imgs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        xray_imgs.append(img[y:y+h, x:x+w])  # Crop the X-ray based on bounding box

    # Ensure we have two sub-images (left and right X-rays)
    img1, img2 = xray_imgs

    # Create masks for each sub-image
    def create_mask(sub_img):
        # The red color in the image has high values for the red channel and low values for the other channels
        lower_red = np.array([150, 0, 0])  # Lower bound for red
        upper_red = np.array([255, 100, 100])  # Upper bound for red
        mask = cv2.inRange(sub_img, lower_red, upper_red)
        return mask // 255  # Normalize mask to binary (0 and 1)

    mask1 = create_mask(img1)
    mask2 = create_mask(img2)

    # Calculate Dice coefficient
    intersection = np.sum(mask1 * mask2)
    dice = (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-6)
    
    return dice

def check_dice(image_path1, image_path2):
    # Load images using PIL and convert to RGB
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # Convert images to NumPy arrays
    np_img1 = np.array(img1)
    np_img2 = np.array(img2)

    # Convert images to PyTorch tensors and move to GPU
    tensor1 = torch.from_numpy(np_img1).to('cuda')
    tensor2 = torch.from_numpy(np_img2).to('cuda')

    # Create binary mask where image is not greyscale
    mask1 = (tensor1[:, :, 0] != tensor1[:, :, 1]).to(torch.uint8)
    mask2 = (tensor2[:, :, 0] != tensor2[:, :, 1]).to(torch.uint8)

    if torch.equal(mask1,mask2):
        return 1.0

    # Calculate the Dice coefficient
    intersection = torch.sum(mask1 & mask2)
    total = torch.sum(mask1) + torch.sum(mask2)

    dice = (2.0 * intersection) / (total + np.finfo(float).eps)

    return dice.item()

def project_and_compare(ct, y1, y2, base_url='http://0.0.0.0:8000/v1', 
                        size=512, organ=None, temp_dir='random',
                        text1=BodyRegionText, 
                        textOrganPresent=ComparisonText2Figs, 
                        textOrganNotPresent=NoOrganSimple, 
                        summarize=CompareSummarize2Figs,axis=1,
                        checkDice=True,window='bone'):
    
    if organ is None:
        for org in organ_list:
            if org in y1[y1.rfind('/')+1:]:
                organ=org
                break
        if organ is None:
            raise ValueError('Organ not found in annotation filenames. Please set it explicitly using the organ parameter.')
        print('Organ inferenced automatically from the annotation filename:', organ)

    # Project the CT scan
    if temp_dir=='random':
        temp_dir='./tmp'+str(random.randint(0,10000))
    os.makedirs(temp_dir, exist_ok=True)

    prj.overlay_projection_fast(pid='y1_bone', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y1,
                           ct_only=False,window=window,axis=axis)
    
    prj.overlay_projection_fast(pid='y2_bone', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y2,
                           ct_only=False,window=window,axis=axis)
    
    if checkDice:
        dice=check_dice(os.path.join(temp_dir,'y1_bone_overlay_window_'+window+'_axis_1_liver.png'),
                        os.path.join(temp_dir,'y2_bone_overlay_window_'+window+'_axis_1_liver.png'))
        print('2D dice coefficient between 2 projections on axis '+str(axis)+':',dice)
        if dice>0.9:
            print('The projections are too similar, skipping the comparison and returning 1.5. Try another axis or ct compare slices (holes?).')
            return 1.5

    prj.overlay_projection_fast(pid='ct', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y1,
                           ct_only=True,window=window,axis=axis)
    
    prj.overlay_projection_fast(pid='y1_organs', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y1,
                           ct_only=False,window='organs',axis=axis)

    
    prj.overlay_projection_fast(pid='y2_organs', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y2,
                           ct_only=False,window='organs',axis=axis)
    
    #prj.create_composite_image(temp_dir, organ,axis=axis,window=window)
    prj.create_composite_image_2figs(temp_dir, organ,axis=axis,window=window)
    
    #API call to LLM
    raise ValueError('I must change below to compare with 2 images and skeleton.')
    ct=os.path.join(temp_dir,'ct_ct_window_'+window+'_axis_1_liver.png')
    y1=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y1.png')
    y2=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y2.png')
    #consider that the correct answer is 2
    answer=Prompt3MessagesLMDeploy(
                    img1=ct,img2=twoImages,img3=twoImages,
                    base_url=base_url,size=size,
                    text1=text1,
                    textOrganPresent=textOrganPresent,
                    textOrganNotPresent=textOrganNotPresent,
                    summarize=summarize,
                    organ=organ)
    
    print('Answer:',answer)
    shutil.rmtree(temp_dir)
    return answer

    

def SystematicComparison3MessagesLMDeploy6Figs(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText6Figs+LiverDescription, 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize6Figs, organ='liver'):
        answers=[]
        outputs={}

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            twoImages=anno.replace('overlay_','2BoneImages')
            anno=anno.replace('overlay_','6Images')
            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=anno,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)

def SystematicComparison3MessagesLMDeploy2FigsOld(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText2Figs+LiverDescription, 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize2Figs, organ='liver',
                    mode='Tissue',save_memory=True):
        answers=[]
        outputs={}

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            if mode=='bone':
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                twoImages=anno.replace('overlay_','2TissueImages')
            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=twoImages,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ,save_memory=save_memory)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)


def SystematicComparison3MessagesLMDeploy1Fig(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText1Fig+LiverDescription, 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize1Fig, organ='liver',
                    mode='bone'):
        answers=[]
        outputs={}

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            if mode=='bone':
                superpos=anno.replace('overlay_','superpositionBone_')
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                superpos=anno.replace('overlay_','superpositionTissue_')
                twoImages=anno.replace('overlay_','2TissueImages')

            
            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=superpos,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)


SinglePrompt=("The images I am sending are frontal projections of one CT scan. They are not CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray does. All 4 images are projections of the same 3D CT scan. \n"
              "Here is a full description of each image:\n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI),  tuned for better bone visualization. The image is superposed with liver overlay 1.\n"
"Image 2: frontal projection of a CT scan using a less wide window (400 UI),  tuned for better abdominal organ visualization. The image is also superposed with liver overlay 1 (like image 1).\n"
"Image 3: frontal projection of the CT scan using a wide window (2000 UI),  tuned for better bone visualization. The image is superposed with liver overlay 2.\n"
"Image 4: frontal projection of a CT scan using a less wide window (400 UI),  tuned for better abdominal organ visualization. The image is also superposed with liver overlay 2 (like image 3).\n"
"In all images, a blue R indicates the right side of the human body, and a green L indicates the left side. Answer the following questions in a stepwise manner:\n"
"Q1- Look carefully, tell me which body region the CT scan represents and where its limits are. Provide a detailed list of all organs usually contained in this region.\n"
"Q2- Based on your answer to question Q1, should the %(organ)s be present in this image? Answer with just ‘yes’ or ‘no’.. If your answer is ‘no’, fill the following template, substituting  _ by Yes or No, and add it to the end of your answer:\n"
"\n"
"Q2 = _\n"
"\n"
"- If your answer to Q2 is ‘yes’, continue to question 3 (below). \n"
"- If your answer to Q2 is ‘no’, ignore Q3 and tell me which overlay has smallest amount of red color. Then, fill the template substituting _ by 1 or 2 (do not fill the template if your answer to Q2 is yes): \n"
"'The best overlay is Overlay _.'\n"
"\n"
"Q3- The %(organ)s region in the images should be marked in red, using an overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")
DescribeLiver=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
"a) The liver is a large organ, with triangular or wedge-like shape.\n"
"b) The liver is located in the upper right quadrant of the abdomen (right is indicated with a blue R in the figures), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. It spans across the midline, partially extending into the left upper quadrant. Looking at the bones images 1 and 3, you can easily identify if the liver should appear in the images or not. The liver is not in the pelvis.\n"
"c) The liver is a single structure. \n"
"d) The liver position is primarily under the rib cage.\n"
"e) If the CT scan does not include the liver region (for example, a scan showing just the pelvis), the correct overlay is actually the one showing no red region (or as little as possible).\n")
SinglePromptSepImages=("The images I am sending are frontal projections of one CT scan. They are not CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray does. In all images, a blue R indicates the right side of the human body, and a green L indicates the left side. All 3 images are projections of the same 3D CT scan. \n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI) for better bone visualization, you can use this image to better understand which body region the CT scan encompasses.\n"
"Image 2: frontal projection of a CT scan using a narrow window (400 UI) for better organ visualization. The image is also superposed with %(organ)s overlay 1, in red.\n"
"Image 3: the same frontal projection of a CT scan as image 1 and 2, but superposed with %(organ)s overlay 2.\n"
"Answer the following questions in a stepwise manner:\n"
"Q1- Look carefully, tell me which body region the CT scan represents and where its limits are. Provide a detailed list of all organs usually contained in this region.\n"
"Q2- Based on your answer to question Q1, should the %(organ)s be present in this image? Answer with just ‘yes’ or ‘no’.. If your answer is ‘no’, fill the following template, substituting  _ by Yes or No, and add it to the end of your answer:\n"
"\n"
"Q2 = _\n"
"\n"
"- If your answer to Q2 is ‘yes’, continue to question 3 (below). \n"
"- If your answer to Q2 is ‘no’, ignore Q3 and tell me which overlay has smallest amount of red color. Then, fill the template substituting _ by 1 or 2 (do not fill the template if your answer to Q2 is yes): \n"
"'The best overlay is Overlay _.'\n"
"\n"
"Q3- The %(organ)s region in the images should be marked in red, using an overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")
compareSummarizeSepImages=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                "A LVLM like you compared the 2 overlays by analyzing images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overlay is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "The text is:\n")

def Prompt2MessagesLMDeploy(img, base_url='http://0.0.0.0:8000/v1', size=512,
                    text1=SinglePrompt,
                    summarize=CompareSummarize, organ='liver'):
    if organ=='liver':
        organDescription=DescribeLiver

    _, answer = SendMessageLmdeploy([img], base_url=base_url, size=size ,conver=[],
                                    text=text1 % {'organ': organ.replace('_',' ')}+organDescription)
    
    if answer=='':
        return 0.5

    _, answer = SendMessageLmdeploy([], base_url=base_url, size=size, text=summarize+answer, conver=[])

    if 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    elif 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    else:
        return 0.5
    
def Prompt2MessagesMultiImageLMDeploy(img1,img2,img3, base_url='http://0.0.0.0:8000/v1', size=512,
                    text1=SinglePromptSepImages,
                    summarize=compareSummarizeSepImages, organ='liver'):
    
    if organ=='liver':
        organDescription=DescribeLiver

    _, answer = SendMessageLmdeploy([img1,img2,img3], base_url=base_url, size=size ,conver=[],
                                    text=text1 % {'organ': organ.replace('_',' ')}+organDescription)
    
    if answer=='':
        return 0.5

    _, answer = SendMessageLmdeploy([], base_url=base_url, size=size, text=summarize+answer, conver=[])

    if 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    elif 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    else:
        return 0.5
    
def SystematicComparison2MessagesLMDeployMultiImage(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=224,
                            text1=SinglePromptSepImages, 
                    summarize=compareSummarizeSepImages, organ='liver'):
    answers=[]

    outputs={}

    for target in os.listdir(pth):
        if 'ct_axis_1' not in target or 'BestIs' in target:
            continue
        print(target)

        ct=os.path.join(pth,target)
        overlay1=ct.replace('ct_','overlay_bad_tissue')
        overlay2=ct.replace('ct_','overlay_better_tissue')

        #consider that the correct answer is 2
        answer=Prompt2MessagesMultiImageLMDeploy(
                        img1=ct,
                        img2=overlay1,
                        img3=overlay2,
                        base_url=base_url,size=size,
                        text1=text1,
                        summarize=summarize,
                        organ=organ)
        print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
        if answer==1:
            answers.append(0)
        elif answer==2:
            answers.append(1)
        outputs[target]=answer

        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
    print('Answer:',answers)
    for k,v in outputs.items():
        print(k,v)
    
def SystematicComparison2MessagesLMDeploy(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=SinglePrompt, 
                    summarize=CompareSummarize, organ='liver'):
    answers=[]

    outputs={}

    for target in os.listdir(pth):
        if 'overlay_axis_1' not in target or 'BestIs' in target:
            continue
        print(target)

        anno=os.path.join(pth,target)
        #consider that the correct answer is 2
        answer=Prompt2MessagesLMDeploy(
                        img=anno,
                        base_url=base_url,size=size,
                        text1=text1,
                        summarize=summarize,
                        organ=organ)
        print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
        if answer==1:
            answers.append(0)
        elif answer==2:
            answers.append(1)
        outputs[target]=answer

        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
    print('Answer:',answers)
    for k,v in outputs.items():
        print(k,v)

def SystematicComparison2MessagesLMDeploySimple(pth,base_url='http://0.0.0.0:8000/v1', 
                                          size=512,
                            text1=SinglePrompt, 
                    summarize=CompareSummarize, organ='liver'):
    answers=[]

    outputs={}

    for target in os.listdir(pth):
        if 'overlay_axis_1' not in target or 'BestIs' in target:
            continue
        print(target)

        anno=os.path.join(pth,target)
        #consider that the correct answer is 2
        answer=Prompt2MessagesLMDeploy(
                        img=anno,
                        base_url=base_url,size=size,
                        text1=text1,
                        summarize=summarize,
                        organ=organ)
        print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
        if answer==1:
            answers.append(0)
        elif answer==2:
            answers.append(1)
        outputs[target]=answer

        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
    print('Answer:',answers)
    for k,v in outputs.items():
        print(k,v)
        



def Prompt3MessagesQwen(img1, img2, model, processor,process_vision_info,
                    text1=BodyRegionText, 
                    textOrganPresent=ComparisonText+LiverDescription, 
                    textOrganNotPresent=NoOrganText, summarize=CompareSummarize, organ='liver'):
    
    conversation, answer = SendMessageQwen([img1], model, processor, process_vision_info=process_vision_info, text=text1 % {'organ': organ.replace('_',' ')})

    
    AnswerNo=('no' in answer.lower()[answer.lower().rfind('q2'):answer.lower().rfind('q2')+15])
    
    if AnswerNo:
        text2 = textOrganNotPresent % {'organ': organ.replace('_',' ')}
    else:   
        text2 = textOrganPresent % {'organ': organ.replace('_',' ')}

    conversation, answer = SendMessageQwen([img1,img2], model, processor, process_vision_info=process_vision_info, text=text2, conversation=conversation)

    conversation, answer = SendMessageQwen([], model, processor, process_vision_info=process_vision_info, text=summarize+answer, conversation=[])

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5
    

    
def SystematicComparison3MessagesQwen(pth,model,processor,process_vision_info,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText+LiverDescription, 
                    textOrganNotPresent=NoOrganText, 
                    summarize=CompareSummarize, organ='liver'):
        answers=[]

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            #consider that the correct answer is 2
            answer=Prompt3MessagesQwen(
                            img1=clean,img2=anno,
                            model=model,processor=processor,
                            process_vision_info=process_vision_info,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')






























StomachDescriptionEDGPT="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is a hollow organ with a J-shaped or curved sac-like structure. It is not a solid organ, but it is not an amorphous blob.
b) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs, between the esophagus and the small intestine.
c) The stomach is situated between the esophagus (above) and the small intestine (below), and is in close proximity to the spleen and pancreas."""
#GPT generated - 65 ACC

StomachDescriptionEDV1="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure. 
b) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
c) The stomach red overlay should be a single shape, with a smooth surface and no holes."""


StomachDescriptionEDV2="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure. 
b) The stomach is NOT an amorphous blob and it does NOT have sharp edges.
c) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
d) The stomach red overlay should be a single smooth shape."""
#0.75 Accuracy

StomachDescriptionEDV3="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The shape of the stomach red overlay should resemble an inverted L or inverted J or a curved sac-like structure. Its lower end should point to the right side of the body (left side of the image, which is oriented like an AP X-ray). 
b) The stomach is NOT an amorphous blob.
c) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
d) The stomach red overlay should be a single smooth shape."""
#0.62 accuracy

StomachDescriptionEDV2="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure. 
b) The stomach is NOT an amorphous blob and it does NOT have sharp edges.
c) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
d) The stomach red overlay should be a single smooth shape."""


StomachDescriptionEDV3="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure.
b) Its curve does not point up.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
e) The stomach red overlay should be a single smooth shape."""
#0.75 accuracy, erros actually make sense

StomachDescriptionEDVX="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure.
b) Its curve does not point up.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
e) The stomach red overlay should be a single structure."""
#0.81 accuracy, erros actually make sense

StomachDescriptionEDV5="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure.
b) Its curve does not point up.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is mainly located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs. Its bottom part extends towards the right side of the body (left side of the image), and it may cross the spine.
e) The stomach red overlay should be a single structure."""
#75% acc

StomachDescriptionEDV6="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or L-shaped or curved sac-like structure.
b) Its curve does not point up.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is mainly located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs. Its lower part can cross the midline and enter the right side of the human body.
e) The stomach red overlay should be a single structure."""
#75% acc

StomachDescriptionEDMultiStage="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach red overlay can have the follwoing shapes, depending on digestion: 
- Empty stomach: a small J-shaped or curved sac-like structure.
- Early digestion: the J-shape may still be visible, but it becomes bigger, more rounded and distended. It may look rounder at the top.
- Mid-digestion: the stomach may have an hourglass shape, with a contraction in the middle and two bulging areas—one at the top and one at the bottom. It may also look like a rounded, distended sac, with some sesemblance of its J shape.
b) Its curve does not point up, when the curve is clearly visible it resembles a J.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is mainly located in the upper part of the abdomen, just below the diaphragm.
e) The stomach red overlay should be a single structure."""
#68%

StomachDescriptionEDV8="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The stomach is an organ with a J-shaped or curved sac-like structure. The stomach red overlay should be a single structure.
b) Wrong shapes: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
c) Location: The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#68%

StomachDescriptionEDV9="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is an organ with a J-shaped or curved sac-like structure.
b) Its curve does not point up.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
e) The stomach red overlay should be a single structure."""
#68%

StomachDescriptionEDV10="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The stomach is a J-shaped or curved sac-like structure. When empty, it is small and compact, and may look folded. During digestion, it becomes larger and more rounded but still maintains some of its J-shape; it may become rounded at the top, at the middle, at the bottom or it also take on an hourglass shape with a contraction in the middle and two bulging areas.
b) Its curve does not point up.
c) The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
d) The stomach is located in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs.
e) The stomach red overlay should be a single structure."""
#52%



StomachDescriptionEDV11="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The stomach is an organ with a J shape or curved sac-like structure. With a J shape, tts curve points down. During some time of diggestion, it may be more inflated in different regions, or have a hourglass shape.
c) Defined and smooth shape: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges, gaps or fragmentation.
e) Unity: The stomach red overlay should be a single structure. Not being a single structure is an obvious overlay error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""



StomachDescriptionEDV12="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The stomach is an organ with a J shape or curved sac-like structure, with a curvature flexing downwards. During some time of diggestion, it may be more inflated in different regions, or have a hourglass-like shape.
b) Shape 2: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges and gaps.
c) Unity: The stomach red overlay should be a single structure. If the overlay shows disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""

StomachDescriptionEDV12="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The stomach is an organ with a J shape or curved sac-like structure, with a curvature flexing downwards. During some time of diggestion, it may be more inflated in different regions, or have a hourglass-like shape.
b) Shape 2: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges and gaps.
c) Unity: The stomach red overlay should be a single structure. If the overlay shows multiple disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""

StomachDescriptionEDV13="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The stomach is an organ with a J shape or curved sac-like structure, with a curvature flexing downwards. During some time of diggestion, it may be more inflated in different regions, or have a hourglass-like shape.
b) Shape 2: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges and gaps.
c) Unity: The stomach red overlay should be a single structure. If the overlay shows disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#Above 80% accuracy



StomachDescriptionEDBad="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach can resemble the letter J, an inverted letter L, or a sac with a downwards curvature. During some time of digestion, it may be more inflated in different regions, or have a hourglass-like shape.
b) Shape 2: The stomach is NOT an amorphous blob. It is also NOT a random shape with random sharp edges or internal holes.
c) Unity: The stomach red overlay should be a single structure. If the overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#65% accuracy

StomachDescriptionEDVXX="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach can resemble the letter J, an inverted letter L, or a sac with a downwards curvature. During some time of digestion, it may be more inflated in different regions, or have a hourglass shape.
b) Shape 2: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges or internal holes.
c) Unity: The stomach red overlay should be a single structure. If the overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#100% accuracr with 5 samples rejected as "do not know".


StomachDescriptionEDV14="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach can resemble the letter J, or a flipped letter L, or a sac with a downwards curve. During some time of diggestion, it may be more rounded in different regions, or have a hourglass shape with two bulges.
b) Shape 2: The stomach is NOT an amorphous blob. It is also NOT a random shape with random edges or internal holes.
c) Unity: The stomach is a single connected structure. If the red overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#75% accuracy

StomachDescriptionEDV15="""The image I am sending now, we can call it Image 2, is a frontal projection of a CT scan. It is not a CT slice, we have transparency and can see through the entire body, like a X-ray. The left side of the image represents the right side of the human body, it looks like an AP (anterior-to-posterior) X-ray. The stomach region in the image should be marked in red, using an overlay. However, I am not sure if the red overlay correctly or incorrectly marks the stomach. Check if the red region is coherent with the expected shape and location of a stomach, and analyze potential mistakes, if any.When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach can resemble the letter J, an inverted letter L, or a sac with a downwards curvature. The stomach can resemble a J-shape, flipped L-shape, or a sac-like structure with a downward curve. It is not expected to be perfectly uniform but should follow these general contours. It may be more rounded in different regions, or have a hourglass shape with 2 round areas.
b) Shape 2: The stomach is NOT a random shape with random edges or internal holes.
c) Unity: The stomach is a single connected structure. If the red overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#100% acc, 4 rejections

StomachDescriptionEDV16="""The image I am sending now, we can call it Image 2, is a frontal projection of a CT scan. It is not a CT slice, we have transparency and can see through the entire body, like a X-ray. The left side of the image represents the right side of the human body, it looks like an AP (anterior-to-posterior) X-ray. The stomach region in the image should be marked in red, using an overlay. However, I am not sure if the red overlay correctly or incorrectly marks the stomach. Check if the red region is coherent with the expected shape and location of a stomach, and analyze potential mistakes, if any.When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach resembles the letter J, an inverted letter L, or a sac with a downwards curvature. It is not perfectly uniform, it may be more rounded in different areas, or have a hourglass shape with 2 rounded areas.
b) Shape 2: The stomach is NOT a random shape with random edges or internal holes.
c) Unity: The stomach is a single connected structure. If the red overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#88% accuracy 6 rejections

StomachDescriptionEDV17="""The image I am sending now, we can call it Image 2, is a frontal projection of a CT scan. It is not a CT slice, we have transparency and can see through the entire body, like a X-ray. The left side of the image represents the right side of the human body, it looks like an AP (anterior-to-posterior) X-ray. The stomach region in the image should be marked in red, using an overlay. However, I am not sure if the red overlay correctly or incorrectly marks the stomach. Check if the red region is coherent with the expected shape and location of a stomach, and analyze potential mistakes, if any.When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach resembles the letter J, an inverted letter L, or a sac with a downwards curvature. It is not perfectly uniform, but should follow these general contours. It may be more rounded in different areas, or have a hourglass shape with 2 rounded areas.
b) Shape 2: The stomach is NOT a random shape with random edges or internal holes.
c) Unity: The stomach is a single connected structure. If the red overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper left quadrant of the abdomen (right side of the figure, like an AP X-ray), just below the diaphragm. It lies mostly under the left ribs."""
#82% 2 prompts non conservative

StomachDescriptionEDV18="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach can resemble the letter J, an inverted letter L, or a sac with a downwards curvature. During some time of digestion, it may be more inflated in different regions, or have a hourglass shape.
b) Shape 2: The stomach is NOT a random shape with random edges or internal holes.
c) Unity: The stomach is a single connected structure. If the red overlay shows multiple structures or small disconnected parts, it has a big error.
d) Location: The stomach is located mainly in the upper abdomen, just below the diaphragm. It lies mostly under the ribs."""
#82% 2 prompts non conservative, 4 rejections
#cnservative: would give 100% with 11 rejections!
#If you force yes or no answers, you get 100% accuracy with 9 rejections


StomachDescriptionEDV20="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The shape of the stomach red overlay should resemble the letter J, an inverted letter L, or a sac with a downwards curvature. It is not expected to be perfectly uniform but should follow these general contours. It may be more rounded in different regions, or have a hourglass shape with two bulges.
c) Unity: The stomach red overlay should be a single connected structure. Big errors overlay are: showing multiple structures, disconnected parts, or internal voids.
d) Location: The stomach red overlay should be located mainly in the upper abdomen, starting just below the diaphragm. It lies mostly under the ribs."""
#Force yes or no answers: 72% accuracy, 4 rejections


GallbladderDescriptionEDV0="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The gallbladder is a small, pear-shaped organ.
b) The gallbladder is located in the upper right quadrant of the abdomen (left side of the figure, like an AP X-ray).
c) The gallbladder sits near the lower edge of the liver and may overlap with the liver in frontal projections.
d) The gallbladder is a single structure."""
#GPT generated
PancreasDescriptionEDV0="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is an elongated, flat organ with a tadpole-like or elongated S-shape.
b) The pancreas is located in the upper abdomen, behind the stomach. It extends horizontally from the right side of the abdomen (left side of the image, like an AP X-ray) to the left side (right side of the image).
c) The head of the pancreas is situated on the right side of the abdomen (left side of the image), nestled in the curve of the duodenum, while the tail extends toward the spleen on the left side (right side of the image), near the left kidney."""
#GPT generated

PancreasDescriptionEDV1="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is an elongated, flat organ with a tadpole-like shape, with its tail pointing to the left side of the body (right side of the image, like an AP X-ray).
a) The pancreas head is its thickest part.
b) The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
c) The pancreas is a single smooth shape, it is not fragmented."""

PancreasDescriptionEDV2="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is an elongated, flat organ with a tadpole-like shape.
a) The pancreas head is its thickest part. Thus, the pancreas is thicker in the left side of the image/right side of the body (the image is oriented like an AP X-ray).
b) The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
c) The pancreas is a single smooth shape, it is not fragmented."""



PancreasDescriptionEDV3="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is an elongated, flat organ with a tadpole-like shape. Its can also look like a curved ribbon or a comma. The pancreas is significantly thicker in one of its sides. 
a) The pancreas head is its thickest part. This thickest part points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray.
b) The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
c) The pancreas is a single smooth shape, it is not fragmented, and it does not have very sharp edges."""

PancreasDescriptionEDV4="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is an elongated, flat organ with a tadpole-like shape. Its can also look like a curved ribbon or a comma. It has a thicker head at one side and a thinner tail at the other. The head is notably thicker than the tail.
a) The pancreas head is its thickest part. This thickest part points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray.
b) The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
c) The pancreas is a single smooth shape, it is not fragmented, and it does not have very sharp edges."""

PancreasDescriptionEDV5="""When evaluating and comparing the overlays, individually adress each of the following points:
a) The pancreas is an elongated organ with a tadpole-like shape. It has a thicker head at one side and a thinner tail at the other.
b) The pancreas head is its thickest part. This thickest part points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray.
c) The pancreas is located in the upper abdomen near the bottom of the rib cage.
d) The pancreas is a single smooth shape, it is not fragmented, and it does not have very sharp edges.
e) Position: the pancreas is mainly horizontal, with a curve pointing upwards. The head can be at the tail height or lower, the tail cannot go much lower than the head."""
#bad


PancreasDescriptionEDv6="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is an elongated, flat organ with a tadpole-like shape. Its can also look like a curved ribbon or a comma. The pancreas is significantly thicker in one of its sides. 
b) The pancreas head is its thickest part. This thickest part points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray.
b) The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
c) The pancreas is a single smooth shape, it is not fragmented.
d) The head of the pancreas usually sits lower than the tail."""
#bad


PancreasDescriptionEDV7="""When evaluating and comparing the overlays, consider the following anatomical information:
a) The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage.
b) The pancreas is an elongated, flat organ with a tadpole-like shape. It is mostly horizontal, with a curve pointing up. 
c) The pancreas head is its thickest part. This thickest part points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray.
d) The head of the pancreas usually sits lower than the tail.
e) The pancreas is a single smooth shape and it does not have very sharp edges."""
#horrible

PancreasDescriptionEDV8="""When evaluating and comparing the overlays, consider the following anatomical information:
a) Shape: The pancreas is an elongated organ with a tadpole-like shape. The pancreas head is its thickest part and points to the left side of the image, which is the right side of the body because the image is oriented like an AP X-ray. Besides its head, the pancreas is thin.
b) Position: The pancreas is located in the upper abdomen, behind the stomach and near the bottom of the rib cage. The head of the pancreas is usually lower than the tail.
C) Smoothness: The pancreas is a single smooth shape and it does not have very sharp edges."""
#good


DescendingAortaDescriptionEDV0 = (
 "The aorta runs parallel to the spine, which usually appears as a vertical line along the midline of the body. It can be curved when the spine is curved. The descending aorta is the part of the aorta that extends from the upper chest to the lower abdomen. The arota does NOT start on the lower chest, the aorta does NOT start at the diaphragm level. \n"#The aorta is tubular and continuous, it does not have any gap in the middle. \n"
)
#70%
DescendingAortaDescriptionEDV1 = (
 "The aorta runs parallel to the spine, which usually appears as a vertical line along the midline of the body. It can be curved when the spine is curved. The descending aorta is the part of the aorta that extends from the upper chest to the lower abdomen. The arota does NOT start on the lower chest, and the aorta does NOT start at the diaphragm level. The aorta is tubular and continuous, it does not have any gap in the middle. \n"
)
#50%


DescendingAortaDescriptionEDV2 = (
 "The aorta runs parallel to the spine, which usually appears as a vertical line along the midline of the body. It can be curved when the spine is curved. The descending aorta is the part of the aorta that extends from the upper chest to the lower abdomen. Therefore, the top end of the aorta shoud not visible in this image, being up from its upper boundary. The arota does NOT start on the lower chest, and the aorta does NOT start at the diaphragm level. The aorta does not have any gap or missing part, it is continuous. \n"
)
#60%

DescendingAortaDescriptionEDV3=(
    "The aorta runs vertically along the midline of the body, typically appearing parallel to the spine in "
    "frontal CT projections. It closely follows the spine's curve if the spine is curved. "
    "The descending aorta starts in the upper chest and extends into the lower abdomen. "
    "It begins just after the arch of the aorta in the upper chest, continuing down toward the diaphragm, "
    "but does NOT start at or below the diaphragm. "
    "The aorta is a continuous tubular structure without gaps. "
    "In radiology images, correct annotations should reflect this smooth, continuous path from the chest "
    "down to the abdomen, properly aligned with the spine."
)#GPT


DescendingAortaDescriptionEDV4 = (
 "The aorta runs parallel to the spine, which usually appears as a vertical line along the midline of the body. It can be curved when the spine is curved. The aorta should start at the upper chest. The arota does NOT start on the lower chest, the aorta does NOT start at the diaphragm level. \n"#The aorta is tubular and continuous, it does not have any gap in the middle. \n"
)


DescendingAortaDescriptionEDV5 = (
 "The aorta runs parallel to the spine and very close to it, which usually appears as a vertical line along the midline of the body. It can be curved when the spine is curved. The descending aorta is the part of the aorta that extends from the upper chest to the lower abdomen. The arota does NOT start on the lower chest, the aorta does NOT start at the diaphragm level. \n"#The aorta is tubular and continuous, it does not have any gap in the middle. \n"
)
#70% with skeleton view

DescendingAortaDescriptionEDV6 = (
 "The descending aorta  (the part of the aorta without the aortic arch) runs parallel to the spine, which usually appears as a vertical line along the midline of the body. A correct descending aorta overlay can be central and excatcly over the spine, partially over the spine, or right next to the spine. It can be curved when the spine is curved. The descending aorta does NOT start on the lower chest, and the descending aorta does NOT start at the diaphragm level, it starts HIGHER. Since this image does not reach the top of the heart, we want the red overlay to extent AS HIGH AS POSSIBLE. \n"#The aorta is tubular and continuous, it does not have any gap in the middle. \n"
)

DescendingAortaDescriptionEDV7 = (
    "The descending aorta (the part of the aorta without the aortic arch) runs parallel to the spine, which appears as a vertical "
    "or slightly curved line in the midline of the body. A correct descending aorta overlay should be central, exactly over the spine, "
    "partially over the spine, or adjacent to the spine. The overlay may follow any curvature of the spine if present. The descending aorta "
    "starts in the upper chest, extending as high as possible in the image, as this projection does not show the descending aorta starting point. The descending aorta does NOT start near the diaphragm or the lower ribs, "
    "which is much lower in the body. The aorta is a continuous tubular structure."
)
#50%
DescendingAortaDescriptionEDV8 = (
 "Closely follow the following instructions in evauating the image:\n"
 "1- Do you see a red overlay in the image? \n"
 "2- Does the red overlay resemble a long, vertical line approximatelly in the center of the body? It may be slightly curved if the spine is curved. \n" #The aortic arch is not present.
 "3- Does the red line touch the top of the image? it MUST. This is very important, check carefully. \n"
 "4- Does the red line extend as far as the end of the lumbar spine (if it is visible in the image)? \n"
)
#50 %