import sys
import codecs
import json

# 입력 및 출력 인코딩 설정
sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())


data = sys.stdin.read()

# JSON 데이터를 딕셔너리로 역직렬화
jsonData = json.loads(data)
print('데이터',jsonData) 

##################################################################################















##################################################################################

sys.stdout.write('hate')
sys.stdout.flush()