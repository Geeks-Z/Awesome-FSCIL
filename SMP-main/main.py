import json
import argparse
from trainer import train
def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--base_model_path', type=str, default='', help='model path for base session')
    # backbone_type: vit_base_patch16_224, vit_base_patch16_224_in21k
    parser.add_argument('--backbone_type', type=str, default='vit_base_patch16_224', help='model path for base session')
    parser.add_argument('--tag', type=str, default='default', help='tag for this training.')
    parser.add_argument('--mid_layers', type=int, nargs='+', default=[8, 9, 10], help='List of mid layers') # for px 7
    parser.add_argument('--common_param', type=str, help='tag for this training.')
    parser.add_argument('--common_param2', type=str, help='tag for this training.')
    parser.add_argument('--merge_weight', type=str, default='0.5', help='tag for this training.')
    parser.add_argument('--special_flag', action='store_true', help='flag tag for special flow.') # 特殊逻辑flag
    parser.add_argument('--special_tag', default='special', help='filename tag for special flow.') # 特殊逻辑标记
    parser.add_argument('--plot', action='store_true', help='plot flag for landscape.')
    return parser

if __name__ == '__main__':
    main()
