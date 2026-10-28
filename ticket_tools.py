import json

TICKETS = {
    "TK-001": {
        "ticket_id": "TK-001",
        "description": "办公室多台电脑无法连接 Wi-Fi，有线网络可以使用。",
        "device": "办公区无线接入点 AP-01",
        "processing_status": "待处理",
    },
    "TK-002": {
        "ticket_id": "TK-002",
        "description": "电脑能够正常上网，但打开报表软件后立即闪退。",
        "device": "办公电脑 PC-02",
        "processing_status": "处理中",
    },
}


def get_ticket(ticket_id):
    """根据工单编号查询模拟工单，只读取数据，不修改工单。"""

    # 第一步：检查编号的数据类型
    if not isinstance(ticket_id, str):
        return {
            "status": "invalid_input",
            "message": "工单编号必须是字符串。",
        }

    # 第二步：去掉编号前后的空格,并且无视大小写
    ticket_id = ticket_id.strip().upper()

    if not ticket_id:
        return {
            "status": "invalid_input",
            "message": "工单编号不能为空。",
        }

    # 第三步：查找工单
    ticket = TICKETS.get(ticket_id)

    if ticket is None:
        return {
            "status": "not_found",
            "message": "未找到该工单，请核对工单编号。",
        }

    # 第四步：返回查询结果
    return {
        "status": "ok",
        "ticket": ticket.copy(),
    }


if __name__ == "__main__":
    ticket_id = input("请输入工单编号，例如 TK-001：")

    result = get_ticket(ticket_id)

    print(json.dumps(result, ensure_ascii=False, indent=2))