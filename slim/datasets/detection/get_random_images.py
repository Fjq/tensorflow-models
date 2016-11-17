raise RuntimeError('deprecated, use extract_bboxes.py with --randomize-bboxes instead')
"""
"""
from __future__ import division
from __future__ import print_function

import os, tarfile, argparse

from PIL import Image

from extract_bboxes import iter_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_tar', help='images')
    parser.add_argument('bbox_folder',
                        help='root directory for .xml bounding boxes')
    parser.add_argument('wnid', help='wnid you don\'t want in there')
    parser.add_argument('--present-wnid',
                        help='wnid that you want in the image (comma-separated)')
    parser.add_argument('output_dir', help='where we will write the .JPEGs')
    parser.add_argument('--nb-output-subdirs',
                        help='so that we don\'t have too many files in a given folder',
                        default=10, nargs='?', type=int)
    parser.add_argument('--min-distinct-wnid',
                        help='only keeps images with at least that many distinct wnid',
                        default=1)
    args = parser.parse_args()
    print(args)

    tar = tarfile.open(args.image_tar)
    basename = os.path.splitext(os.path.basename(args.image_tar))[0]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    i = 0
    for data in iter_boxes(args.bbox_folder, args.wnid, keep_wnid=False,
                           min_distinct_wnid=int(args.min_distinct_wnid),
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
        out_fname = os.path.join(args.output_dir,
                                    str(i % args.nb_output_subdirs),
                                    'image_%i.JPEG' % i)
        out_dir = os.path.dirname(out_fname)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        img.save(out_fname)
        i += 1
        if i % 100 == 0:
            print('%i done' % i)
