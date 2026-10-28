import json
from copy import deepcopy


# 项目自编的模拟处理规范，用于开发和测试。
RULES = {
    "网络": {
        "rule_id": "RULE-NET-001",
        "version": "1.0",
        "source": "项目自编模拟规范",
        "title": "网络连接故障初步排查",
        "checks": [
            "确认影响范围：单台设备还是多台设备。",
            "分别确认有线网络和无线网络是否可用。",
            "记录故障开始时间，以及此前是否发生配置变更。",
            "收集连接失败提示和相关日志，供进一步分析。",
        ],
        "constraints": [
            "不能仅凭无法连接就认定接入点损坏。",
            "重启设备、修改配置等操作需要人工确认和授权。",
        ],
    },
    "软件": {
        "rule_id": "RULE-SOFT-001",
        "version": "1.0",
        "source": "项目自编模拟规范",
        "title": "软件启动或运行异常初步排查",
        "checks": [
            "记录软件名称、版本和操作系统版本。",
            "确认异常发生在启动时还是执行特定操作时。",
            "收集错误提示、崩溃日志和复现步骤。",
            "确认是否有其他设备出现相同问题。",
        ],
        "constraints": [
            "不能仅凭软件闪退就认定是网络故障。",
            "卸载软件、清理数据等操作需要人工确认和授权。",
        ],
    },
}


def get_handling_rule(category):
    """根据故障类别，读取对应的模拟处理规范。"""

    if not isinstance(category, str):
        return {
            "status": "invalid_input",
            "message": "故障类别必须是字符串。",
        }

    category = category.strip()

    if not category:
        return {
            "status": "invalid_input",
            "message": "故障类别不能为空。",
        }

    rule = RULES.get(category)

    if rule is None:
        return {
            "status": "not_found",
            "message": "没有找到该类别的处理规范。",
        }

    return {
        "status": "ok",
        "category": category,
        "rule": deepcopy(rule),
    }


if __name__ == "__main__":
    category = input("请输入故障类别，例如 网络 或 软件：")

    result = get_handling_rule(category)

    print(json.dumps(result, ensure_ascii=False, indent=2))