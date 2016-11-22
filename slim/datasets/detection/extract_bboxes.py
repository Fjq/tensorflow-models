"""
Crops the ILSVRC2014 Detection Challenge images using the bounding boxes
for a given synset
"""
from __future__ import division
from __future__ import print_function

import os, tarfile, argparse
from xml.etree import cElementTree
import random

from PIL import Image


class Bboxes(object):
    def __init__(self):
        self.bboxes = []

    def append(self, bbox):
        self.bboxes.append(bbox)

    def __len__(self):
        return len(self.bboxes)


def iter_boxes(bbox_folder,
               required_wnid=None,
               forbidden_wnid=None,
               bbox_wnid=None,
               min_distinct_wnid=1):
    """
    `min_distinct_wnid`: images with less than `min_distinct_wnid` distinct
    wnid will be ignored
    `image_has_wnid` : list of wnid that must be in an image for it to be
    considered (at least one of them is fine)
    """
    for root, dirs, files in os.walk(bbox_folder):
        print('in ' + root)
        files = [f for f in files if f.endswith('.xml')]
        for f in files:
            f = os.path.join(root, f)
            try:
                data = cElementTree.parse(f).getroot()
            except cElementTree.ParseError:
                print('error parsing ' + f)
                continue
            folder = data.find('folder').text
            filename = data.find('filename').text
            source = data.find('source').find('database').text
            size = data.find('size')
            size = [int(size.find(what).text) for what in ['height', 'width']]

            bboxes = Bboxes()

            has_required_wnid = False
            has_forbidden_wnid = False

            unique_wnid = set()
            for obj in data.findall('object'):
                name = obj.find('name').text
                unique_wnid.add(name)

                if required_wnid is None \
                        or has_required_wnid \
                        or name in required_wnid:
                    has_required_wnid = True

                if forbidden_wnid is not None \
                        and name in forbidden_wnid:
                    has_forbidden_wnid = True
                    break

                # if we care about whhich bbox we use, then continue if it's not
                # the right one
                if bbox_wnid is not None \
                        and name not in bbox_wnid:
                    continue

                bbox = obj.find('bndbox')
                bbox = [int(bbox.find(what).text) for what in
                        'ymin', 'ymax', 'xmin', 'xmax']

                bbox[0] /= size[0]
                bbox[1] /= size[0]
                bbox[2] /= size[1]
                bbox[3] /= size[1]

                # inspired from google's `process_bounding_boxes.py` script
                # https://github.com/tensorflow/models/blob/master/inception/inception/data/process_bounding_boxes.py
                bbox[0] = min(bbox[0], bbox[1])
                bbox[1] = max(bbox[0], bbox[1])
                bbox[2] = min(bbox[2], bbox[3])
                bbox[3] = max(bbox[2], bbox[3])

                # bigger bounding box
                h = bbox[1] - bbox[0]
                w = bbox[3] - bbox[2]
                bbox[0] -= h/10
                bbox[1] += h/10
                bbox[2] -= w/10
                bbox[3] += w/10

                bbox[0] = max(bbox[0], 0.)
                bbox[1] = min(bbox[1], 1.)
                bbox[2] = max(bbox[2], 0.)
                bbox[3] = min(bbox[3], 1.)

                bboxes.append(bbox)

            # print('----------')
            # print(has_required_wnid)
            # print(has_forbidden_wnid)
            # print(len(bboxes))
            # print(len(unique_wnid))
            # there are a few reasons for which we just discard that image
            if not has_required_wnid \
                    or has_forbidden_wnid \
                    or len(bboxes) == 0 \
                    or len(unique_wnid) < min_distinct_wnid:
                continue

            bboxes.folder = folder
            bboxes.filename = filename
            bboxes.source = source
            yield bboxes


def extract_bbox(img, out_path, bbox):
    width, height = img.size

    up = bbox[0] * height
    low = bbox[1] * height
    left = bbox[2] * width
    right = bbox[3] * width

    box = [left, up, right, low]
    box = [int(round(x)) for x in box]

    img = img.crop(box)

    img.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_tar', help='images')
    parser.add_argument('bbox_folder',
                        help='root directory for .xml bounding boxes')
    parser.add_argument('--bbox-wnid', help='wnid for which you want the bbox')

    parser.add_argument('--required-wnid',
                        help='wnid that you want in the image (comma-separated)')
    parser.add_argument('--forbidden-wnid',
                        help='wnid you *dont* want in the image (comma-separated)')

    parser.add_argument('output_dir', help='where we will write the .JPEGs')
    parser.add_argument('--nb-output-subdirs',
                        help='so that we don\'t have too many files in a given folder',
                        default=10, nargs='?', type=int)

    parser.add_argument('--min-distinct-wnid',
                        help='only keeps images with at least that many distinct wnid',
                        default=1, type=int)

    # parser.add_argument('--randomize-boxes', action='store_true',
                       # help='ignore the bboxes and use 5 random ones')
    parser.add_argument('--full-image', action='store_true',
                       help='no box return the full image')
    parser.add_argument('--random-crop', action='store_true')

    args = parser.parse_args()
    print(args)

    tar = tarfile.open(args.image_tar)
    basename = os.path.splitext(os.path.basename(args.image_tar))[0]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    bbox_wnid = args.bbox_wnid.split(',') if args.bbox_wnid is not None else None
    required_wnid = args.required_wnid.split(',') if args.required_wnid is not None else None
    forbidden_wnid = args.forbidden_wnid.split(',') if args.forbidden_wnid is not None else None

    i = 0
    for data in iter_boxes(args.bbox_folder,
                           bbox_wnid=bbox_wnid,
                           min_distinct_wnid=int(args.min_distinct_wnid),
                           required_wnid=required_wnid,
                           forbidden_wnid=forbidden_wnid):

        try:
            # the train set works with this one
            tar_folder = os.path.join(basename, data.folder + '.tar')
            tar_folder = tarfile.open(fileobj=tar.extractfile(tar_folder))
            fname = os.path.join(data.folder, data.filename + '.JPEG')
        except KeyError:
            # the valid works here (one less .tar file level)
            fname = os.path.join(basename, data.filename + '.JPEG')
            tar_folder = tar

        img = Image.open(tar_folder.extractfile(fname))

        # yes this is pretty hackish
        if args.full_image:
            bboxes = [[0., 1., 0., 1.]]
        elif args.random_crop:
            width, height = img.size
            h = random.uniform(0.05, .6)
            w = h * height / width
            offset_y = random.uniform(0., 1 - h)
            offset_x = random.uniform(0., 1 - w)
            bboxes = [[offset_y, offset_y + h,
                       offset_x, offset_x + w]]

            # left up corner
            # bboxes.append([0,h, 0,w])
            # right up corner
            # bboxes.append([0,h,1-w,1])
            # left bottom
            # bboxes.append([1-h,1,0,w])
            # right bottom
            # bboxes.append([1-h,1,1-w,1])
            # center
            # bboxes.append([(1-h)/2, (1+h)/2,  (1-w)/2, (1+w)/2])
        else:
            bboxes = data.bboxes

        for bbox in bboxes:
            out_fname = os.path.join(args.output_dir,
                                     str(i % args.nb_output_subdirs),
                                     'image_%i.JPEG' % i)
            out_dir = os.path.dirname(out_fname)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            # img.save(out_fname)
            extract_bbox(img, out_fname, bbox)
            i += 1
            if i % 100 == 0:
                print('%i done' % i)
