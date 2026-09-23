"""转换wikipedia-cn-20230720-filtered.json格式的小代码
"""

import json
import os
output_path = "wikipedia-cn-20230720-filtered_formatted.json"
fw = open(output_path, "w", encoding="utf-8")

with open("wikipedia-cn-20230720-filtered.json", "r", encoding="utf-8") as fd:
    data = json.load(fd)
    for item in data:
        line = json.dumps({"text": item["completion"]}, ensure_ascii=False)
        fw.write(line + "\n")
