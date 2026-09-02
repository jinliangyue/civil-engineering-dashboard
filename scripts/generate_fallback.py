"""
PPI 兜底数据生成脚本（已废弃 · 2026-09-02）

历史用途：
    项目早期版本使用此脚本生成 4 个行业 2015-2025 年度 PPI 的兜底估算数据。
    该数据是基于公开 PPI 指数范围的**手工估算值**，不是真实公开数据。

当前状态（2026-09-02 · Phase 3 P0.1 后）：
    此脚本已废弃，保留仅为兼容性。
    当前项目正式数据来源：
        - 月度真实数据：akshare.macro_china_ppi() 间接从国家统计局抓取
        - 不再使用年度手工估算数据

    运行此脚本不会生成任何文件，会打印 deprecation 消息。

作者：十八 · 22 岁土木工程准大四 · 2026 秋招简历项目
"""


def main():
    print('=' * 60)
    print('DEPRECATION: generate_fallback.py 已废弃')
    print('=' * 60)
    print('原因：')
    print('  - 4 行业 2015-2025 年度 PPI 数据原本是基于公开范围的手工估算')
    print('  - 不是真实公开数据，不能用于严格实验')
    print('  - 当前项目已删除相关年度数据')
    print('')
    print('当前正式数据源：')
    print('  - 月度 PPI：akshare.macro_china_ppi() 间接从国家统计局抓取')
    print('  - 文件：data/raw/工业PPI_全国月度_2015-2025.csv')
    print('  - 共 132 个真实月度点（2015-01 ~ 2025-12）')
    print('')
    print('详细见：Phase 2 v3.1 实验设计 / docs/PROJECT_STATUS.md')
    print('=' * 60)


if __name__ == '__main__':
    main()
