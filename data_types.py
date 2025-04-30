from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import json
import os
import re


@dataclass
class TopConceptsResult:
    neu_id_cur: dict
    neu_id_prev: dict
    parent_group_id: int
    parent_group_imgs: dict
    children_group_imgs: dict
    respond: Optional[Dict[str, str]] = field(default_factory=dict)
    child_group_ids: Optional[List[int]] = field(default_factory=list)
    weights: Optional[Dict[int, float]] = field(default_factory=dict)
    group_fm_pair: Optional[Dict[str, int]] = field(default_factory=dict)
    group_label: Optional[Dict[str, int]] = field(default_factory=dict)
    
    def unpack(self) -> Tuple[dict, dict, int, Optional[List[int]], Optional[Dict[int, float]]]:
        return self.neu_id_cur, self.neu_id_prev, self.parent_group_id, self.child_group_ids, self.weights, self.parent_group_imgs, self.children_group_imgs, self.respond, self.group_fm_pair, self.group_label
    
    def to_dict(self):
        return {
            "neu_id_cur": self.neu_id_cur,
            "neu_id_prev": self.neu_id_prev,
            "parent_group_id": self.parent_group_id,
            "child_group_ids": self.child_group_ids,
            "weights": self.weights,
            "parent_group_imgs": self.parent_group_imgs,
            "children_group_imgs": self.children_group_imgs,
            "respond": self.respond,
            "group_fm_pair": self.group_fm_pair,
            "group_label": self.group_label
        }
    
    def has_children(self) -> bool:
        return bool(self.child_group_ids)

    def get_all_ids(self) -> List[int]:
        return [self.parent_group_id] + (self.child_group_ids or [])

    def summary(self) -> str:
        return f"Parent Group: {self.parent_group_id}, Children: {self.child_group_ids or []}"


class LLMGroupParser:
    def __init__(self, top_concepts_list: TopConceptsResult):
        self.top_concepts_list = top_concepts_list
        self.weights = top_concepts_list.weights
        self.gpt_outputs = top_concepts_list.respond
        self.parent_group_imgs = top_concepts_list.parent_group_imgs
        self.children_group_imgs = top_concepts_list.children_group_imgs
        self.neu_id_cur = top_concepts_list.neu_id_cur
        self.neu_id_prev = top_concepts_list.neu_id_prev
        
        self.group_fm_pair = {}
        self.group_label = {}
        
        self.group_hierarchy = {}
        self.group_metadata = {}
        self.group_captions = {}
        self.feature_evolution = {}
        self.feature_weights = {}

    def parse(self) -> Dict[str, int]:
        for key, text in self.gpt_outputs.items():
            groups = key.split("_")
            parent = groups[0]
            children = groups[1:]

            # initial hierarchy
            self.group_hierarchy.setdefault(parent, []).extend(children)

            # parse Common Features
            feature_matches = re.findall(r"- Group (\w+) Common Features: <<(.+?)>>", text)
            # merge parent + children
            combined_imgs = OrderedDict()
            combined_imgs.update(self.parent_group_imgs)
            combined_imgs.update(self.children_group_imgs)
            
            combined_labels = OrderedDict()
            combined_labels.update(self.neu_id_cur)
            combined_labels.update(self.neu_id_prev)

            for (group, description), (fm_id, imgs), (fm_id2, label) in zip(feature_matches, combined_imgs.items(), combined_labels.items()):
                assert fm_id==fm_id2, "Feature map not match!"
                label = int(label[0]) if isinstance(label, list) else int(label)
                self.group_metadata[group] = {
                    "description": description.strip(),
                    "images": imgs,
                    "label": label,
                    "feature_map_id": fm_id
                }
                self.group_fm_pair[group] = fm_id
                self.group_label[group] = label

            # parse Feature Evolution
            evolution_matches = re.findall(r"- Group (\w+): <<(.+?)>>", text.split("- Feature Evolution:")[-1])
            for child_group, relation in evolution_matches:
                self.feature_evolution.setdefault(parent, {})[child_group] = relation.strip()
                self.feature_weights.setdefault(parent, {})[child_group] = self.weights[int(child_group.split('G')[-1])]

            # parse Caption, key is full group key（parent + all children）
            caption_match = re.search(r"- Caption: <<(.+?)>>", text)
            if caption_match:
                full_group_key = '_'.join([parent] + children)
                self.group_captions[full_group_key] = caption_match.group(1).strip()
                
        # LXGX and feature map id pair
        return self.group_fm_pair, self.group_label  

    def save_all(self, save_dir=""):
        os.makedirs(save_dir, exist_ok=True)
        for name, data in {
            "group_hierarchy": self.group_hierarchy,
            "group_metadata": self.group_metadata,
            "group_captions": self.group_captions,
            "feature_evolution": self.feature_evolution,
            "feature_weights": self.feature_weights,
        }.items():
            with open(os.path.join(save_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
 