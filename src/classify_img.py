"""
#####################################################################################################################

    Module to classify image content

#####################################################################################################################
"""

import      os
import      cv2
import      torch
from        ultralytics     import YOLO

IMG_PATH    = "../imgs"
# IMG_PATH    = "../imgs_all/imgs_alice"

THRESHOLD   = 0.4   # Minimum fraction of the image that should be covered by people

# Load the YOLOv8 model (pretrained on COCO)
# Use "yolov8s.pt" for better accuracy
model       = YOLO( "yolov8n.pt" )

def check_person_img( fimg ):
    """
    Check if people occupy the majority of the image.

    params:
        fimg:   [str] Path to the image file

    return:
        [bool]  true if people occupy the majority of the image
        [float] coverage of people in image
    """
    # Load image
    img     = cv2.imread( fimg )
    if img is None:
        print( f"Could not load image: {fimg}" )
        return False, 0

    img_height, img_width   = img.shape[ :2 ]
    image_area              = img_width * img_height

    # Run inference
    results     = model( fimg, verbose=False )

    person_area = 0
    for r in results:
        for box in r.boxes:
            cls = int( box.cls[ 0 ] )   # Class ID
            if cls == 0:                # 'Person' class in COCO dataset
                x1, y1, x2, y2  = box.xyxy[0]  # Bounding box
                bbox_area       = ( x2-x1 ) * ( y2-y1 )
                person_area     += bbox_area

    # Compute ratio of image occupied by people
    coverage    = person_area / image_area
    coverage_b  = coverage >= THRESHOLD
    return coverage_b, coverage


def check_person_imgs( fpath ):
    """
    Process all images in a folder and determine if people are the main content.

    params:
        fpath:  [str] Path to the folder containing images

    return:
        [dict]  key: news id
                value:  [bool]  true if people occupy the majority of the image
                        [float] coverage of people in image
    """
    results = {}
    cnt     = 0
    tot     = len( os.listdir( fpath ) )
    
    for f in os.listdir( fpath ):
        print( f"Processing image {cnt+1} out of {tot}" )
        cnt += 1
        if f.lower().endswith( ( '.jpg', '.jpeg', '.png', '.bmp', '.gif' ) ):
            i               = os.path.join( fpath, f )
            cb, c           = check_person_img( i )
            ff              = f.split( '.' )[ 0 ]
            results[ ff ]   = ( cb, c )
    return results


#####################################################################################################################

if __name__ == '__main__':
    results = check_person_imgs( IMG_PATH )
    ct, cf  = 0, 0
    for r in sorted( results ):
        if results[ r ][ 0 ]:
            ct += 1
        else:
            cf += 1
        print( f"{r}:\t\t{results[ r ][ 0 ]}\t{results[ r ][ 1 ]:.2f}" )
    print( f"Total: {ct} images with people, {cf} images without." )
