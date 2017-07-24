#!/usr/bin/env python3

import os
import sys
import numpy as np
import argparse
from kaffe import KaffeError, print_stderr

def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')


def convert(def_path, caffemodel_path, data_output_path, code_output_path,
            phase, framework):
    try:
        if framework == 'tensorflow':
            from kaffe.tensorflow import TensorFlowTransformer
            transformer = TensorFlowTransformer(def_path, caffemodel_path,
                                                phase=phase)
        elif framework == 'theano':
            from kaffe.theano import TheanoTransformer
            transformer = TheanoTransformer(def_path, caffemodel_path,
                                            phase=phase)
        elif framework == 'tf_nchw':
            from kaffe.tf_nchw import TF_NCHW_Transformer
            transformer = TF_NCHW_Transformer(def_path, caffemodel_path,
                                              phase=phase)
        else:
            raise NotImplementedError('Not implemented target to convert')

        print_stderr('Converting data...')
        if caffemodel_path is not None:
            data = transformer.transform_data()
            print_stderr('Saving data...')
            with open(data_output_path, 'wb') as data_out:
                np.save(data_out, data)
        if code_output_path:
            print_stderr('Saving source...')
            with open(code_output_path, 'w') as src_out:
                src_out.write(transformer.transform_source())
        print_stderr('Done.')
    except KaffeError as err:
        fatal_error('Error encountered: {}'.format(err))
    except NotImplementedError as err:
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('def_path', help='Model definition (.prototxt) path')
    parser.add_argument('target', help='The target framework to convert',
                        choices=['tensorflow', 'theano', 'tf_nchw'])
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    parser.add_argument('--data-output-path', help='Converted data output path')
    parser.add_argument('--code-output-path', help='Save generated source to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')

    args = parser.parse_args()
    validate_arguments(args)
    convert(args.def_path, args.caffemodel, args.data_output_path, args.code_output_path,
            args.phase, args.target)

if __name__ == '__main__':
    main()
