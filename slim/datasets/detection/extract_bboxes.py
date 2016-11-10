"""
Crops the ILSVRC2014 Detection Challenge images using the bounding boxes
for a given synset
"""
from __future__ import division
from __future__ import print_function

import sys, os, tarfile, argparse
from xml.etree import cElementTree

from PIL import Image


class Bboxes(object):
    def __init__(self):
        self.bboxes = []

    def append(self, bbox):
        self.bboxes.append(bbox)

    def __len__(self):
        return len(self.bboxes)


def iter_boxes(bbox_folder, wnid, keep_wnid=True, min_distinct_wnid=1,
               present_wnid=None):
    """
    If `keep_wnid` == False, then we will *ignore* images that contain at
    least one element of that `wnid`
    `min_distinct_wnid`: images with less than `min_distinct_wnid` distinct
    wnid will be ignored
    `present_wnid` : list of wnid that must be in an image for it to be
    considered
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
            skip_image = False
            has_present_wnid = False
            unique_wnid = set()
            for obj in data.findall('object'):
                name = obj.find('name').text
                unique_wnid.add(name)
                if present_wnid is None \
                        or has_present_wnid \
                        or name in present_wnid:
                    has_present_wnid = True

                if keep_wnid:
                    if name != wnid:
                        continue
                # ignore that wnid
                else:
                    if name == wnid:
                        skip_image = True
                        break

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

                bbox[0] = max(bbox[0], 0.)
                bbox[1] = min(bbox[1], 1.)
                bbox[2] = max(bbox[2], 0.)
                bbox[3] = min(bbox[3], 1.)

                bboxes.append(bbox)

            # there are a few reasons for which we just discard that image
            if skip_image \
                    or not has_present_wnid \
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
    parser.add_argument('wnid', help='wnid you want to search for')
    parser.add_argument('--present-wnid',
                        help='wnid that you want in the image (comma-separated)')
    parser.add_argument('output_dir', help='where we will write the .JPEGs')
    parser.add_argument('--nb-output-subdirs',
                        help='so that we don\'t have too many files in a given folder',
                        default=10, nargs='?', type=int)
    parser.add_argument('--min-distinct-wnid',
                        help='only keeps images with at least that many distinct wnid',
                        default=1, type=int)
    args = parser.parse_args()
    print(args)

    tar = tarfile.open(args.image_tar)
    basename = os.path.splitext(os.path.basename(args.image_tar))[0]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    i = 0
    for data in iter_boxes(args.bbox_folder, args.wnid, min_distinct_wnid=int(args.min_distinct_wnid),
                           present_wnid=args.present_wnid.split(',')):

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
        # print(fname)
        for bbox in data.bboxes:
            out_fname = os.path.join(args.output_dir,
                                     str(i % args.nb_output_subdirs),
                                     'image_%i.JPEG' % i)
            out_dir = os.path.dirname(out_fname)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            extract_bbox(img, out_fname, bbox)
            i += 1
            if i % 100 == 0:
                print('%i done' % i)
