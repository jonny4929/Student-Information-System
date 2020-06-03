import cv2
def clip_boxes(bbox,im_size):
    bbox[bbox<0]=0
    bbox[bbox[:,2]>im_size[0]]=im_size[0]
    bbox[bbox[:,3]>im_size[1]]=im_size[1]
    bbox=bbox[bbox[:,0]<bbox[:,2]]
    bbox=bbox[bbox[:,1]<bbox[:,3]]
    return bbox

def warp_affine(image, points, scale=1.0):
        eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        dy = points[1][1] - points[0][1]
        dx = points[1][0] - points[0][0]
        # 计算旋转角度
        angle = cv2.fastAtan2(dy, dx)
        rot = cv2.getRotationMatrix2D(eye_center, angle, scale=scale)
        rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))
        return rot_img