import argparse
import logging
import time

import cv2
import numpy as np
import re
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# num_squats = 3
fps_time = 0
initial_state = True
inter_state = False

i_LHip = None
i_RHip = None
i_LShoulder = None
i_RShoulder = None

bodyparts_index_dict = {
"Nose" : 0,
"Neck" : 1,
"RShoulder" : 2,
"RElbow" : 3,
"RWrist" : 4,
"LShoulder" : 5,
"LElbow" : 6,
"LWrist" : 7,
"RHip" : 8,
"RKnee" : 9,
"RAnkle" : 10,
"LHip" : 11,
"LKnee" : 12,
"LAnkle" : 13,
"REye" : 14,
"LEye" : 15,
"REar" : 16,
"LEar" : 17,
"Background" : 18
}
def giveIndex(h,s):
    
    return (float(h[re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]-1:re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]+3]),float(
        h[re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]+5:re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]+9])) if h.find("BodyPart:"+str(bodyparts_index_dict[s])+"-(")!=-1 else -1

def calculateCredits_squats(h,c):
    h = str(h)
    if h == None:
        return
    global inter_state
    global initial_state
    global i_LHip
    global i_RHip
    global i_LShoulder
    global i_RShoulder

    # delta = 0.05
    if initial_state == True:
        #initial
        print("here")
        print(h)
        i_LHip = giveIndex(h,"LHip")
        i_RHip = giveIndex(h,"RHip")
        i_LShoulder = giveIndex(h,"LShoulder")
        i_RShoulder = giveIndex(h,"RShoulder")
        if i_LHip!=-1:
            initial_state = False
        print("initial_state = ",initial_state)
        return 0 if c == None else c

    #intermediate
    inter_LShoulder = giveIndex(h,"LShoulder")
    inter_RShoulder = giveIndex(h,"RShoulder")
    delta = 0.2

    if i_LHip != None and i_RHip != None and inter_LShoulder != None:
        print("*******************inside second ********************")
        print("i_LHip[1] = ",i_LHip,"delta = ", delta, "inter_LShoulder[1]", inter_LShoulder)
        if type(inter_LShoulder)!=int:
            if i_LHip[1] - delta <= inter_LShoulder[1]: #or i_LHip + delta >= inter_LShoulder:
                print("here 2")
                inter_state = True
    else:
        logger.debug("Error: Value not found", i_LHip, i_RHip, i_LShoulder)
        return -1
    
   #final 
    final_LShoulder = giveIndex(h,"LShoulder")
    final_RShoulder = giveIndex(h,"RShoulder") 
    if final_LShoulder != None and final_RShoulder != None:
        print("*************inside check*********************")
        print(final_LShoulder, i_LShoulder)
        if type(final_LShoulder)!=int:
            if final_LShoulder[1] - 0.2 <= i_LShoulder[1] and final_LShoulder[1] + 0.05 >=i_LShoulder[1]: 
                print("here 3")
                if inter_state:
                    inter_state = False
                    initial_state = True
                    c=c+1
                    print("count = ",c)
                    return c

            
    else:
        logger.debug("Error: Value not found", final_LShoulder, final_RShoulder)
        return -1
    return 0 if c==None else c
    # Initial LShoulder, RShoulder 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    c=0
    while c<10:
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        logger.debug(humans)
        print("***********************************************Count in main before ******************************",c)        
        c=calculateCredits_squats(humans,c)
        print("***********************************************Count in main after ******************************",c)
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(image,"COUNT: %i"%(c),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cam.release()
    cv2.destroyAllWindows()

# def giveIndex(h,s):
#     return (h[re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]-1:re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]+3],
#         h[re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]+5:re.search("(BodyPart:"+str(bodyparts_index_dict[s])+"-\()\w+",h).span()[1]+9])

# def calculateCredits_squats(h,c):
#     h = str(h)
    
#     if initial_state == True:
#         #initial
#         i_LHip = giveIndex(h,"LHip")
#         i_RHip = giveIndex(h,"RHip")
#         i_LShoulder = giveIndex(h,"LShoulder")
#         i_RShoulder = giveIndex(h,"RShoulder")
#         initial_state = False
    

#     #intermediate
#     inter_LShoulder = giveIndex(h,"LShoulder")
#     inter_RShoulder = giveIndex(h,"RShoulder")
#     delta = 0.09
#     if i_LHip != None and i_RHip != None and inter_LShoulder != None:
#         if i_LHip - delta <= inter_LShoulder: #or i_LHip + delta >= inter_LShoulder:
#             inter_state = True
#     else:
#         logger.debug("Error: Value not found", i_LHip, i_RHip, i_LShoulder)
    
    
#    #final 
#     final_LShoulder = giveIndex(h,"LShoulder")
#     final_RShoulder = giveIndex(h,"RShoulder") 
#     delta = 0.05
#     if final_LShoulder != None and final_RShoulder != None:
#         if final_LShoulder - delta <= i_LShoulder: 
#             if inter_state:
#                 inter_state = False
#                 initial_state = True
#                 c+=1

#     else:
#         logger.debug("Error: Value not found", final_LShoulder, final_RShoulder)
#     # Initial LShoulder, RShoulder 



