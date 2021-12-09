ID: 20212047
Name: 김예찬

Evo-AI HW4: Genetic Programming (Find the symbolic formula!)

[프로그램 실행 방법]
[0] Install the python libraries as in requirements.txt
    pip install -r requirements.txt
[1] Run the code
    python src_main.py --file_path="data(gp)/data-gp1.txt" --pop_size=500 --gen_size=500 --tree_depth_max=15
    - file_path에는 제공한 데이터셋의 경로를 전달합니다.
    - pop_size는 maximum population size를 의미합니다.
    - gen_size는 maximum generation size를 의미합니다.
    - tree_depth_max는 예측하고자 하는 함수 f(x)의 Symbolic Tree (계산식 트리)의 depth 최댓값을 의미합니다.

[결과물 확인 방법]
[0] 프로그램 실행이 종료되면, data(gp) 폴더 아래에 다음과 같은 파일이 형성됩니다.
    1) "data-gp* - best_fitness_score: xxx.png" -> 실제 정답과 "Best Symbolic Formula의 예측 값"을 비교함.
    2) "best-gp* - formula.txt" -> "Best Symbolic Formula" 결과가 저장됨. (트리 구조로 저장함.)
    3) "data-gp* - fitnesses.png" -> 유전 알고리즘의 매 generation 마다의 fitness score 결과 값을 시각화함.

[기타 유의사항]
[0] 계산식 트리 구조의 경우, 직접 구현하는 대신 Python Anytree 라이브러리를 추가로 사용하였습니다.

[1] 계산식 트리의 연산자 노드의 경우, 크게 2개 유형의 연산자를 고려하였습니다.
    - 이항 연산자 (bi_operators): ADD (+), SUB (-), MUL (X)
    - 단항 연산자 (uni_operators): SIN, COS, TAN, LOG (+abs), SQRT (+abs)
      -> 지수 함수 (exp) 역시 고려할 수는 있으나, 큰 값이 exp()에 전달될 경우,
         Infty 오류가 발생하여 사용하지 않았습니다. (구현 편의)
      -> 또, Log와 Sqrt의 경우, 음수 인자를 전달받을 경우, NaN 오류가 발생하므로,
         인자를 abs() 함수로 항상 양수로 변환하여 전달받게끔 제한 (즉, log(abs(x))와 같이 제한)을 두었습니다. (구현 편의)

[2] 계산식 트리의 피연산자 노드의 경우, 다음과 같습니다.
     - 변수 'x': symbol (data_gp*.txt 파일의 'x' 컬럼 값이 대입될 변수)
     - 상수: 적절한 상수 값들 (e.g. -0.38461538461538414, 0.384615384615385, 1.1538461538461542, 1.9230769230769234, ...]
            -> min_constant와 max_constant를 셋팅하면, min_constant 이상 max_constant 이하의
               일정한 등간격의 실수 값 상수들이 자동으로 생성됩니다.
     자세한 사항은 src_utils.py / generate_operands() 함수를 참조하시기 바랍니다.

[3] 계산식 트리의 루트 노드는 항상 ADD (+), SUB (-), MUL (x) 3개의 이항 연산자 중 하나가 되도록 제한을 두었습니다.
    이 루트 노드의 목적은 왼쪽 서브트리와 오른쪽 서브트리를 자연스럽게 연결(Linking)해주는 것입니다.

[4] Fitness 함수는 "제 모델의 예측 값"과 "실제 정답 값 (data-gp*.txt의 'y' 컬럼 값)" 사이의 절댓값 오차 (L1 Error)를
    그 메트릭으로 채택하여 구현하였습니다. 제 모델의 훈련 목표는 고로 Fitness 값을 최대한 최소화해야 하는 것입니다.
    (물론 L2 Loss나 L1과 L2의 Hybrid Loss를 사용하는 방법도 가능은 하지만, 간단하게 L1 Loss를 사용하였습니다.)
    자세한 사항은 src_utils.py / get_fitness_score() 함수를 참조하시기 바랍니다.

[5] Selection의 경우, Soft-tourament 방식을 채택하였습니다.
    자세한 사항은 src_utils.py / tournament_selection() 함수를 참조하시기 바랍니다.

[6] Cross-over의 경우, 루트노드의 왼쪽 서브트리의 서브트리와 오른쪽 서브트리의 서브트리 간의 교배가 이루어지도록
    구현하였습니다. 구현의 편의를 위해, 이항 연산자를 부모로 하는 서브트리 들만 교환이 되도록 구현되어 있습니다.
    자세한 사항은 src_utils.py / crossover() 함수를 참조하시기 바랍니다.

[7] Mutation의 경우, 간단하게 임의의 연산자 노드를 선택하여, 그 임의의 연산자를 다른 연산자 노드로 치환하는 방식으로
    변종이 발생하도록 구현하였습니다.
    구축한 계산식 트리가 망가지지 않도록,
    1) 이항 연산자는 반드시 이항 연산자로,
    2) 단항 연산자는 반드시 단항 연산자로 변이되도록 제한을 두었습니다.
    또, 랜덤하게 경우에 따라는 피연산자가 다른 피연산자로 변이되도록 제한을 두었습니다.

Reference (참고한 논문)
1. "Genetic programming and non-linear multiple regression techniques to predict back-break in blasting operation,"
    Roohollah et al, Engineering and Computers (2016) 32:123-133.