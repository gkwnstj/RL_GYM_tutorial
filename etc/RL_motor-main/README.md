# 시뮬링크를 활용한 파이썬 예제

# 개요
- time step 단위로 simulink를 simulation으로 활용할 수 있다.
    - time step 단위로 변수를 observate하여 python으로 가져올 수 있다. 
    - python에서 simulink로 parameter (input과 같은)을 입력(수정)할 수 있다.
- simulatino이 시작 할 때, 초기값으로서 parameter를 수정할 수 있다.
- simularion을 제어 'start', 'pause', 'continue' 등을 할 수 있다.

# File List
- main.py
    - main function
- simManager.py -> SimManager (Class)
    - ``__init__()``
        - Simluink 이름 저장
    - ``__get_obs()``
        - 시간 벡터와 current step observation 반환
    - ``__setParameter()``
        - Simulink내 변수 변경
        - block과 변수 이름, 값 (scalar) 입력
    - ``connectMatlab()``
        - Simuilink 연결
    - ``reset()``
        - 초기조건으로 reset
    - ``step()``
        - 한 step 진행
        - ``__get_obs`` 자동 실행 후 결과 반환
    - ``disconnectMatlab()``
- reporter.py -> Reporter (Class)
    - ``__init__()``
    - ``saveRecord()``
        - 시간 벡터와 current observation 저장
    - ``plotRecord()``
        - observation을 시간 축으로 plot

# Work on slx file

1. obeservation으로 변수를 python으로 받기 위해 to workspace 블록을 사용한다.
    -  block parameter의 변수이름을 원하는 이름으로 바꾸되 후에 python의 obsInfo에 기입해야 한다.
    - 하나의 observation당 하나의 블록을 사용
    - '다음 개수의 마지막 점으로 제한'을 1로 조정 (메모리를 아끼기 위해)
    - '저장형식'을 2차원의 배열로 변경
2. 접근하여 수정할 변수의 주소를 파악한다.
    - 시뮬레이션 -> 준비 -> 입력 및 파라미터 조정 -> 파라미터 조정 -> 파라미터로 이동
    - Source는 블록이름, Name은 변수 이름, Value는 현재 설정된 값을 보여준다.

# Work on py file

1. main.py 에서 initial_param 변수에 초기 조건으로 수정할 변수를 block, name, value순으로 입력
    - value는 scalar여야 한다.
    - block name은 정확히 파라미터 조정 탭에서 확인하여야 한다.
2. 매 step 마다 수정할 변수 (입력과 같은)을 같은 방식으로 입력한다.
3. observation할 변수의 이름을 list형식으로 입력한다.

# 실행
1. python main.py로 실행
2. matplot으로 결과 확인

# MAX_TIME은 실제 시뮬링크에서의 tout 기준으로 n번째까지 계산

