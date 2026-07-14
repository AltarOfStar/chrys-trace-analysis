import sys
import json
import urllib.request
import os


def query_session(uuid: str):
    url = "http://lingxi-stats.rnd.huawei.com:8042/api/chrys/query/session"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "filter_key": "uuid",
        "filter_value": uuid
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")

    # 将响应写入 {uuid}.json 到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"{uuid}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data)

    print(f"Response saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python session_request.py <uuid>")
        sys.exit(1)

    uuid = sys.argv[1]
    query_session(uuid)
