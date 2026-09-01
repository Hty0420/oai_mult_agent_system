import json
import pymysql
import os


def fix_database_charset():
    # 获取配置文件路径
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db_config.json')

    # 加载数据库配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 建立数据库连接
    connection = pymysql.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database=config['database'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            # 修改数据库默认字符集
            cursor.execute(
                'ALTER DATABASE {} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci'.format(config['database']))

            # 修改users表字符集
            cursor.execute('ALTER TABLE users CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')

            # 修改conversations表字符集
            cursor.execute('ALTER TABLE conversations CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')

            # 修改messages表字符集
            cursor.execute('ALTER TABLE messages CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')

        connection.commit()
        print("数据库字符集已成功更新为utf8mb4，现在可以存储中文了！")
    except Exception as e:
        print(f"更新字符集时出错: {e}")
    finally:
        connection.close()


if __name__ == "__main__":
    fix_database_charset()