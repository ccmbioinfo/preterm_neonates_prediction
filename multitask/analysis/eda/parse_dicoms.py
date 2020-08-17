import os
import pydicom as dicom
import re
import pandas as pd
import argparse

# pd.options.display.width = 0





# def find_tag(query, tag_df = tags):
#     """
#     Given a query, searches a stacked dataframe of tag info for possible tags and returns them.
#     :param query: (str) search query eg. 'patient'
#     :return: DataFrame of results
#     """
#     try:
#         query = str(query)
#         # results = tags.columns[tags.columns.str.contains(query)]
#         results = tag_df[tag_df.tag_name.str.contains(query)]
#         return results
#     except ValueError:
#         print("Need a string")

def get_total_dcm(root_dir, regex_pattern = '.*'):
    """
    Returns total # of dicoms matching some file pattern in a root directory.
     Defaults to all dicoms in a root directory.
    :param root_dir: (str) root directory to traverse.
    :param regex_pattern: (str) pattern to search file names by, eg. 't1' to match all dicom files with a 't1' in the na,e
    :return: (int) - number of dicoms
    """
    counter = 0
    for dirname, dirnames, filenames in os.walk(root_dir, topdown = True):
        # print(dirname)
        if re.search(regex_pattern, dirname.lower()):
            # if verbose: print("Currently in {}".format(dirname))
            # for each dicom...
            for filename in filenames:
                if '.dcm' not in filename: continue # skip if not a dicom
                counter +=1
    # print(counter)
    return(counter)

# tags   dataelement instance
# key    (value.name value.value)
# (0088, 0140) Storage Media File-set UID          UI: 1.3.12.2.1107.5.99.2.1561.30000009110914374240600000562


def get_dcm_info(root_dir, verbose,
                 regex_pattern = '.*', outname = 'out_dicom',
                 save_to_disk = True):
    """
    Given a directory with folders containing dicom series, traverses all directories and extracts all dicom tags and their values, stored in a dataframe.
    All unique tags are returned as a separate datframe and can be searched by name using find_tag().
    Reorders by slice location and also extracts the image dimensions.

    :param root_dir: (str) - root directory of where your dicoms are stored
    :param verbose: (bool) - whether processing should be printed out to console
    :param regex_pattern: (str) - pattern to parse dicoms by, defaults to all dicoms
    :param outname: (str) - saves to ./output by default, the name of the resulting .csvs
    :param save_to_disk: (bool) - whether to save the results to disk, use False if interacting with the results
    :return: tags, df, tags - 'long' dataframe where each row consists of a tag and its group, element and VR,
                       df - dataframe, where each row consists of a single dicom's tag and info across columns
    """
    df_list = []  # a list where each files md info will be appended to
    md_tags = {}  # for our reference

    for dirname, dirnames, filenames in os.walk(root_dir, topdown = True):
        # print(dirname)
        if re.search(regex_pattern, dirname.lower()):
            if verbose: print("Currently in {}".format(dirname))

            # for each dicom...
            for filename in filenames:
                if '.dcm' not in filename: continue # skip if not a dicom
                fn = os.path.join(dirname, filename)
                if verbose: print("\tProcessing \t: " + fn)
                try:
                    dcm = dicom.read_file(fn) # need a try catch - but it should succeed if it's a dicom, right?
                    md_info = {} # create a new dictionary to hold the header info
                                 # will be appended to a growing list and converted into a dataframe
                    md_info['fn'] = fn # assign file name
                    md_info['pxl_shape'] = dcm.pixel_array.shape
                    # dataset wraps a dictionary, where the keys are (group, element) DICOM tags,
                    # and the value is a DataElement instance
                    # stores DICOM tag, VR, VM, and the tag value

                    # for each tag... extract DataElement instance and add to a dictionary
                    for t in dcm.keys():
                        if (t.group, t.elem) == (0x7fe0, 0x0010): continue # skip if tag refers to the pixel data
                        # print("Key: {}".format(k)) # (eg. 0018, 1315)
                        k = dcm[t] # get the corresponding dataelement (key, value pair) associated with the tag identifier

                        # making column names compliant (all lowercase, symbol-less
                        tag_name = re.sub(r"\[|\]|\'|\(|\)", "", k.name.lower())
                        tag_name = re.sub(r"-|/| ", "_", tag_name)

                        if tag_name == 'slice_location': # slice location
                            v = float(k.value)  # the value corresponding to the current key
                        else:
                            v = str(k.value)
                        # print(new_v)
                        # create a mapping between the key values and a queriable version, will create a new one else replace any old ones
                        md_info[tag_name] = v #

                        # md_info['X%04x_%04x' % (k.group, k.elem)] = md_info[v.name]  # Use both name and group,element syntax
                        md_tags[tag_name] = (
                            t.group, t.element,
                            # '({.04f}_{.04f})'.format(t.group, t.elem),
                            'x(%04x, %04x)' % (t.group, t.elem), # hex representation, add an x so excel/google sheets doesn't automatically convert into negative numbers
                            k.VR)

                    df_list += [md_info] # add the current dicom file info to a list
                except Exception as e:
                    if verbose: print("\tFailed to process \t: " + fn)
                    fail = {'fn': md_info['fn'],
                                          'ReadError': e}  # collect
                    df_list.append(fail)

    relevant_tags = ['fn', 'patients_name', 'acquisition_date',  'series_description', 'pxl_shape', 'sequence_name', 'mr_acquisition_type', 'pixel_spacing',
                     'slice_location', 'scanning_sequence', 'slice_thickness']
    df = pd.DataFrame(df_list)#.columns
    # sort slices
    df.sort_values(['patients_name', 'series_description', 'slice_location'], ascending=True, inplace = True)
    tags = pd.DataFrame.from_dict(md_tags, columns=['group', 'element', 'tag', 'VR'],
                                  orient='index').\
            rename_axis('tag_name').reset_index()
    if save_to_disk:
        out_path = os.path.join(root_dir, 'output', 'dicom_parse')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # all tags and their values
        out_df = os.path.join(out_path, outname + '.csv')
        print('Saving parsed dicom info to {}...'.format(out_df))
        df.to_csv(out_df, index = False)
        print('Done!')

        # just the relevant tags and values
        out_df_slim = os.path.join(out_path, outname + '_slim.csv')
        print('Saving slimmed, parsed dicom info to {}...'.format(out_df_slim))
        df[relevant_tags].to_csv(out_df_slim, index=False)
        print('Done!')

        # the tags
        out_tags = os.path.join(out_path, outname + '_tags.csv')
        print('Saving tag info to {}...'.format(out_tags))
        tags.to_csv(out_tags, index = False)
        print('Done!')
        return tags, df
    else:
        return tags, df

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type = str, required = True, help = 'root directory of dicoms to traverse')
    parser.add_argument('--pattern', type = str, default = '.*', help = 'regex pattern to search for in series description')
    parser.add_argument('--save_to_disk',  default=False, action='store_true', help = 'whether to save to disk or not, saves to root_dir/output/')
    parser.add_argument('--outname', type = str, default = 'out_dicom', help = 'default prefix of output tags and df')
    parser.add_argument('--verbose',  default=False, action='store_true', help = 'verbose output, eg. current series and dicom #')
    return parser

if __name__ == '__main__':

    # get arguments
    args = get_parser().parse_args()

    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))

    # total number of dicoms
    # h = get_total_dcm(root_dir, regex_pattern='.*')
    h = get_total_dcm(args.root_path, regex_pattern = args.pattern) # 127427 (matches dir_files_cts.txt) with no pattern whatsoever, so if our regex is 100% inclusive it must return the same number
    print('Total # of Dicoms: {}'.format(str(h)))
    # get parsed results and tags
    tags, res = get_dcm_info(args.root_path, verbose = args.verbose,
                             regex_pattern = args.pattern,
                             save_to_disk = args.save_to_disk, outname = args.outname)
    print('Finished parsing {}!'.format(args.root_path))

