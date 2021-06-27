from mrcnn import utils
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import skimage
import os

class ForafricDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_small=False, return_coco=True):
        """ Loads dataset released for the crowdAI Mapping Challenge(https://www.crowdai.org/challenges/mapping-challenge)
            Params:
                - dataset_dir : root directory of the dataset (can point to the train/val folder)
                - load_small : Boolean value which signals if the annotations for all the images need to be loaded into the memory,
                               or if only a small subset of the same should be loaded into memory
        """
        self.load_small = load_small
        if self.load_small:
            annotation_path = os.path.join(dataset_dir, "annotation-small.json")
        else:
            annotation_path = os.path.join(dataset_dir, "annotation.json")

        image_dir = os.path.join(dataset_dir, "images")
        print("Annotation Path ", annotation_path)
        print("Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        classIds = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for _class_id in classIds:
            self.add_class("crowdai-mapping-challenge", _class_id, self.coco.loadCats(_class_id)[0]["name"])

        # Register Images
        for _img_id in image_ids:
            assert(os.path.exists(os.path.join(image_dir, self.coco.imgs[_img_id]['file_name'])))
            self.add_image(
                "crowdai-mapping-challenge", image_id=_img_id,
                path=os.path.join(image_dir, self.coco.imgs[_img_id]['file_name']),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                                            imgIds=[_img_id],
                                            catIds=classIds,
                                            iscrowd=None)))

        if return_coco:
            return self.coco

    def load_mask(self, image_id):
        """ Loads instance mask for a given image
              This function converts mask from the coco format to a
              a bitmap [height, width, instance]
            Params:
                - image_id : reference id for a given image

            Returns:
                masks : A bool array of shape [height, width, instances] with
                    one mask per instance
                class_ids : a 1D array of classIds of the corresponding instance masks
                    (In this version of the challenge it will be of shape [instances] and always be filled with the class-id of the "Building" class.)
        """
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
#         #If not a road dataset image, delegate to parent class.
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "crowdai-mapping-challenge":
#             return super(self.__class__, self).load_mask(image_id)
#         print(image_info)
#         # Convert polygons to a bitmap mask of shape
#         # [height, width, instance_count]
#         info = self.image_info[image_id]
#         mask = np.zeros([info["height"], info["width"], len(info["annotations"])],
#                         dtype=np.uint8) 
#         for i, p in enumerate(info["annotations"]):
#             #print(info["path"])
#             segment_length = len(p["segmentation"][0])
#             #print(p["segmentation"][0])
#             all_points_x=[]
#             all_points_y=[]
#             if (segment_length)>2:
#                 for i in range(0,segment_length,2):
#                     all_points_x.append(p["segmentation"][0][i])
#                     all_points_y.append(p["segmentation"][0][i+1])
#             if (segment_length)<=2:
#                 for i in range(0, segment_length):
#                     for j in range(0, len(p["segmentation"][0][i])):
#                         all_points_x.append(p["segmentation"][0][i][j][0])
#                         all_points_y.append(p["segmentation"][0][i][j][1])
#             rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
#         	mask[rr, cc, i] = 1
#         polygons= {'all_points_x': all_points_x, 'all_points_y': all_points_x}
#         mask = np.zeros([info["height"], info["width"], len(polygons)], dtype=np.uint8)
# #         for i, p in
# #             rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
# #         	mask[rr, cc, i] = 1
# #             print("==============") 
# #             print(all_points_x, "for x")
# #             print("--------------")
# #             print(all_points_y, " for y")
# #             print("==============")  
#             # Get indexes of pixels inside the polygon and set them to 1
# #             rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
# #             print("mask.shape, min(mask),max(mask): {}, {},{}".format(mask.shape, np.min(mask),np.max(mask)))
# #             #print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr),np.max(rr)))
# #             #print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc),np.max(cc)))

# #             ## Note that this modifies the existing array arr, instead of creating a result array
# #             ## Ref: https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
# #             rr[rr > mask.shape[0]-1] = mask.shape[0]-1
# #             cc[cc > mask.shape[1]-1] = mask.shape[1]-1

# #             print("After fixing the dirt mask, new values:")        
# #             print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr),np.max(rr)))
# #             print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc),np.max(cc)))

# #                 mask[rr, cc, i] = 1

#         # Return mask, and array of class IDs of each instance. Since we have
#         # one class ID only, we return an array of 1s
#         return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        
        
        
        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-mapping-challenge"

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-mapping-challenge.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation,  image_info["height"],
                                                image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
            #return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MappingChallengeDataset, self).load_mask(image_id)


    def image_reference(self, image_id):
        """Return a reference for a particular image

            Ideally you this function is supposed to return a URL
            but in this case, we will simply return the image_id
        """
        return "crowdai-mapping-challenge::{}".format(image_id)
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']

        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            #rle = mask.merge(rles)           
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
