import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io
from skimage.measure import label, regionprops_table


def read_frame(input_filename, width, height, frame_index):
    """Read frame from raw 2-cam uint8 video file and convert to float32

    Args:
        input_filename: path to raw 2-cam video file
        width: int, frame width in pixels
        height: int, frame height in pixels
        frame_index: int, index of width*height frames to read from input_filename
    Returns:
        (height, width) 0.-1. float32 array
    """
    arr = (
        np.fromfile(
            input_filename,
            dtype="uint8",
            count=width * height,
            offset=frame_index * width * height,
        ).astype("float32")
        / 255
    )
    return arr.reshape(height, width)


def get_background_frame(input_filename, width, height, frame_index, bg_skip_frames=20):
    """Construct background as median of the forward and backward frames

    Args:
        input_filename: path to raw 2-cam video file
        width: int, frame width in pixels
        height: int, frame height in pixels
        frame_index: int, index of width*height frames to read from input_filename
        bg_skip_frames: int, number of width*height frames to skip forward and backward from frame_index
    Returns:
        (height, width) 0.-1. float32 array
        Uses median of forward and backward frame if available, fallback to either frame if unavailable
    """
    try:
        bg_arr1 = read_frame(
            input_filename, width, height, frame_index - bg_skip_frames
        )
    except (OSError, ValueError):
        bg_arr1 = None

    try:
        bg_arr2 = read_frame(
            input_filename, width, height, frame_index + bg_skip_frames
        )
    except (OSError, ValueError):
        bg_arr2 = None

    if bg_arr1 is None and bg_arr2 is None:
        raise IOError(
            "Unable to construct background frame from {}: attempted frame {}+/-{}".format(
                input_filename, frame_index, bg_skip_frames
            )
        )
    elif bg_arr1 is None:
        return bg_arr2
    elif bg_arr2 is None:
        return bg_arr1

    return np.median([bg_arr1, bg_arr2], axis=0)


def detect_2d(
    input_filename,
    output_dir,
    width=2048,
    height=2048,
    save_bgsub=False,
    save_binmask=False,
    bg_skip_ticks=10,
    detection_sigma=5,
    cam1_ymin=1000,
    cam1_ymax=1120,
    cam1_xmin=1106,
    cam1_xmax=1248,
    cam0_ymin=860,
    cam0_ymax=975,
    cam0_xmin=705,
    cam0_xmax=840,
):
    """Read frames from 2-cam video and generate list of detections in each frame

    Args:
        input_filename: path to raw 2-cam video file
        output_dir: path to save output files
        width: int, frame width in pixels
        height: int, frame height in pixels
        save_bgsub: bool, flag to save background-subtracted frames
        save_binamsk: bool, flag to save binary mask frames
        bg_skip_ticks: int, number of ticks to skip forward and backward for background subtraction (1 tick = 2 frames)
        detection_sigma: float, detection significance (# of stddev) for binary mask threshold
        cam{0,1}_{x,y}{min,max}: int, pixel coordinates to selection region of interest for analysis
    Returns:
        csv filename where output detections are saved
    """

    detections = {
        "cam": [],
        "tick": [],
        "region_x": [],
        "region_y": [],
        "frame_x": [],
        "frame_y": [],
    }

    frame_index = 0
    while True:
        # Each "tick" consists of 2 "frames"
        tick = frame_index // 2
        cam = 1 if frame_index % 2 == 0 else 0

        ## Read frame and construct background from leading/lagging frames
        try:
            arr = read_frame(input_filename, width, height, frame_index)
        except ValueError:  # we ran out of frames
            break
        bg_arr = get_background_frame(
            input_filename, width, height, frame_index, bg_skip_ticks * 2
        )
        frame_index += 1

        ## Subtract background
        arr = arr - bg_arr

        ## Slice array to analyze region of interest
        if cam == 1:
            arr = arr[cam1_ymin:cam1_ymax, cam1_xmin:cam1_xmax]
        else:
            arr = arr[cam0_ymin:cam0_ymax, cam0_xmin:cam0_xmax]

        ## Blob detection - find the dark spots
        binary_mask = arr < arr.mean() - detection_sigma * arr.std()
        # label connected components
        blob_labels, num_detections = label(binary_mask, background=0, return_num=True)
        if num_detections == 0:
            print("{} {} No detections".format(tick, cam))
            continue
        # compute centroid coordinates for each blob
        centroids = regionprops_table(blob_labels, properties=["centroid"])

        ## Update detection list
        detections["cam"].extend([cam] * num_detections)
        detections["tick"].extend([tick] * num_detections)
        detections["region_x"].extend(centroids["centroid-1"])
        detections["region_y"].extend(centroids["centroid-0"])
        # convert detection coords from region of interest to full frame coords
        if cam == 0:
            detections["frame_x"].extend(
                [x + cam0_xmin for x in centroids["centroid-1"]]
            )
            detections["frame_y"].extend(
                [y + cam0_ymin for y in centroids["centroid-0"]]
            )
        else:
            detections["frame_x"].extend(
                [x + cam1_xmin for x in centroids["centroid-1"]]
            )
            detections["frame_y"].extend(
                [y + cam1_ymin for y in centroids["centroid-0"]]
            )

        ## Save frames for inspection
        if save_bgsub:
            bgsub_filename = Path(
                output_dir, "frames/cam{}/bgsub/{:04d}.png".format(cam, tick)
            )
            plt.imsave(bgsub_filename, arr, cmap="gray")
        if save_binmask:
            binmask_filename = Path(
                output_dir, "frames/cam{}/binmask/{:04d}.png".format(cam, tick)
            )
            plt.imsave(binamsk_filename, binary_mask, cmap="gray")

        print("{} {} {}".format(tick, cam, num_detections))

        # while loop ends when frame read breaks

    ## Save detections
    det2d_filename = Path(output_dir, "detections_2d.csv")
    df = pd.DataFrame(detections)
    df.to_csv(det2d_filename)
    print("Saved 2d detections to {}".format(det2d_filename))
    return det2d_filename


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", help="Input raw 2-cam video file")
    parser.add_argument(
        "--roi_filename",
        default="region_of_interest.json",
        help="JSON file defining regions of interest",
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path to dir for output files"
    )
    parser.add_argument(
        "--frame_width", type=int, default=2048, help="Input video frame width"
    )
    parser.add_argument(
        "--frame_height", type=int, default=2048, help="Input video frame height"
    )
    parser.add_argument(
        "--save_bgsub",
        action="store_true",
        help="Flag to save background-subtracted frames",
    )
    parser.add_argument(
        "--save_binmask", action="store_true", help="Flag to save binary mask frames"
    )
    parser.add_argument(
        "--bg_skip_ticks",
        type=int,
        default=10,
        help="Number of ticks for leading/lagging background frame to subtract",
    )
    parser.add_argument(
        "--detection_sigma",
        type=int,
        default=5,
        help="Detection significance (# of stddev) for binary mask threshold",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # setup output file directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_bgsub:
        Path(args.output_dir, "frames/cam0/bgsub").mkdir(parents=True, exist_ok=True)
        Path(args.output_dir, "frames/cam1/bgsub").mkdir(parents=True, exist_ok=True)
    if args.save_binmask:
        Path(args.output_dir, "frames/cam0/binmask").mkdir(parents=True, exist_ok=True)
        Path(args.output_dir, "frames/cam1/binmask").mkdir(parents=True, exist_ok=True)

    # get region of interest coordinates
    with open(args.roi_filename) as fp:
        roi_dict = json.load(fp)

    # get 2d detections in each frame
    det2d_filename = detect_2d(
        args.input_filename,
        args.output_dir,
        width=args.frame_width,
        height=args.frame_height,
        save_bgsub=args.save_bgsub,
        save_binmask=args.save_binmask,
        bg_skip_ticks=args.bg_skip_ticks,
        detection_sigma=args.detection_sigma,
        **roi_dict
    )
