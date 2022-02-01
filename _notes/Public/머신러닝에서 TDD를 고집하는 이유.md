---
title : 머신러닝에서 TDD를 고집하는 이유
notetype : feed
date : 01-02-2022
---

얼마 전 회사 내에서 <TDD on Machine Learning>을 주제로 세미나를 진행했다. 테스트 주도 개발의 핵심은 빠른 피드백을 통한 개발 방향 조정이다. 내가 쓴 코드가 정말 목표를 이루는 데 맞는 코드인지 확인하는 것이다. 복잡도가 매우 큰 머신러닝 프로젝트에서 소프트웨어 개발 방법론인 TDD를 사용하여 복잡도를 줄일 수 있다는 것이 내 세미나의 주장이었다. 그리고 질문이 들어왔다. 

> 머신러닝은 일반적인 소프트웨어 개발처럼 정답이 정해져 있고 그것을 해결하는 과정이 아니라, 데이터에서 정답을 추론하는 것에 가까운데, 어떻게 정답이 정해져 있는 TDD를 사용할 수 있나요?

사실 어렵다. 머신러닝/딥러닝의 구조는 매우 복잡하다. 수정한 코드가, 바꾼 모델의 구조가 정확히 어떤 Output을 내는 지 정확히 예측하기 힘들다. 게다가 데이터 전처리, 모델 빌드, 훈련, 평가라는 일련의 과정은 linear하게 흘러가지 않는다. 이번 2021 Pycon KR에서도 코드 수정 없이 AI 모델 배포, 머신러닝 개발 프레임워크 등 머신러닝의 개발 관련 주제가 여러 등장한 이유다. 

그럼에도 불구하고 내가 머신러닝 TDD를 고집하는 이유는 **문서화를 통한 프로젝트 관리**다. 머신러닝 엔지니어라면 바쁜 프로젝트 일정 속에서 Jupyter Notebook 파일이 몇 개나 쌓여있는 경험을 한다. 게다가 거기엔 코드는 마구잡이로 순서가 뒤얽힌 채로 적혀있다. TDD의 테스트 케이스 작성은 이러한 소모적인 개발 시간을 줄여준다.

TDD에서는 테스트를 작성한다. 하나의 테스트는 다음 테스트로 이어지는 이정표(Milestone)가 된다. 테스트 케이스를 작성함으로써 프로젝트의 단계 별 기준을 세운다. 또한 Unit 테스트를 작성하기 위해 계속해서 모듈화가 요구된다. 모듈화는 코드의 재사용성을 높임으로써 데이터 노가다 같은 작업을 줄여준다.

### ML TDD 원칙

1. Git을 이용한 프로젝트/버전 관리
2. 테스트 작성을 통한 명확한 문제 기술

내가 ML에 TDD를 적용하기 위해 노력하면서 꼭 지키려고 하는 두 가지다. Git과 테스트 작성은 유기적인 관계다. 모델을 훈련시키다가도 데이터의 구조를 수정해야할 때면 반드시 목표로하는 테스트를 작성 후에 데이터를 수정한다. **테스트의 Coverage가 프로젝트의 모든 단계에 이르도록 하는 것보다는, 이슈가 생길 때마다 테스트부터 작성하는 것을 목표로 훈련하고 있다.**  여러가지 실험도 해보는 중인데, processing, modeling, pipeline 등의 branch를 만들어 해당하는 분야의 코드를 적을 떄는 checkout해서 개발해 보기도 한다. 아직 까지는 실험 중이다. 

### ML TDD 구조

- src: 모듈화를 위한 소스코드 디렉토리
	- model
	- pipeline
	- preprocess
	- utils
- tests: 테스트 디렉토리
	- conftest
	- test model
	- test dataset
- notes: Jupyter Notebook으로 실험하고, 마크다운 파일에 독립적으로 정리한다
	- ERD.md
	- model.md
- models: 모델 weight 파일 디렉토리
- data:
	- raw
	- processed: raw 데이터를 전처리한 데이터
	- submission: 개별 모델에 맞는 데이터 모음 디렉토리
		- dataset_RandomForestClassifier.csv
	- etc

여러 실험을 통해 잠정적으로 확정한 머신러닝 분야에서 TDD를 적용하는 Repository 구조다. src에 프로젝트를 모듈화하고 이를 tests에서 테스트한다. Jupyter Notebook으로 한 실험 및 테스트 결과를 notes에 저장한다. 아무리 노력을 해봐도 Jupyter Notebook으로 코드와 문서를 동시에 잡는 것은 불가능했다. 한꺼번에 정리하려다 보니 머릿속이 터져버린다. 

테스트에는 [PyTest](https://docs.pytest.org/en/6.2.x/)를 쓴다. 파이썬의 기본 Testing Tool인 UnitTest보다 간편하다. fixture를 decorator로 관리하는 것도 매우 편하다. 내가 작성한 테스트 코드(부끄럽지만)를 보고싶다면 내 [공개 레포](https://github.com/junuMoon/TDD_ML)를 참고하시길 바란다.

```python
@pytest.mark.skip('No difference found')
def test_random_forest_drop_unimportant_feature(dataset, selected_dataset):
    X_train, X_test, y_train, y_test = dataset
    model_orig = RandomForestClassifier(n_estimators=100, oob_score=True)
    model_orig.fit(X_train, y_train)
    score_orig = model_orig.score(X_test, y_test)
    print(f"{model_orig.__class__.__name__} score: {score_orig}")
    print(f"{model_orig.__class__.__name__} oob score: {model_orig.oob_score_}")
    assert score_orig >= 0.8, f"{model_orig.__class__.__name__} failed"

    X_train, X_test, y_train, y_test = selected_dataset
    model = RandomForestClassifier(n_estimators=100, oob_score=True)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{model.__class__.__name__} score: {score}")
    print(f"{model.__class__.__name__} oob score: {model.oob_score_}")
    assert score > score_orig, "Model trained with selected feature score is not better than original"
```

예를 들어 RandomForestClassifier를 테스트해보자. 첫 번째 빌드에서 Feature Importance를 확인하고 0.01이하는 제외하고 훈련을 시켰을 때 성능이 올라갈 지를 확인하고자 한다. 기존의 `dataset` Fixture에서 feature를 제외한 `selected_dataset`이란 fixture를 만든다. 동일한 RandomForestClassifier를 빌드하고 훈련시킨다. 마지막 `assert`문에서 `score`를 비교한다. 

결과적으로 차이가 없었다. 다른 소프트웨어 개발과 달리 이 테스트는 반복해서 할 필요가 없다. decorator로 `skip`을 표시해주고 이유를 단다. 마지막으로 `notes/model.md`에 결과를 적는다.

### Next Step
머신러닝 개발의 소프트웨어 방법론 적용을 익히기 위해서 해보고 싶은 다음 단계들이 있다. 하나하나 씩 적용해보며 좋으면 글로 써봐야지.

1. [Kedro](https://github.com/kedro-org/kedro) - 케드로는 재현 가능하고 유지 보수가 가능하며 모듈화된 데이터 과학 코드를 만들기 위한 오픈 소스 파이썬 프레임워크이다. 그것은 소프트웨어 엔지니어링 모범 사례에서 개념을 빌려 기계 학습 코드에 적용한다.적용된 개념에는 모듈성, 우려 사항 분리 및 버전 관리가 포함된다.
2. [ML Flow](https://mlflow.org) - 기계 학습 주기를 위한 오픈 소스 플랫폼
3. [Build Machine Learning Powered Application](https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X) - 기계 학습(ML)으로 구동되는 애플리케이션을 설계, 구축 및 배포하는 데 필요한 기술을 배우세요. 이 실습 과정을 통해, 당신은 초기 아이디어에서 배포된 제품에 이르기까지 ML 기반 애플리케이션의 예를 구축할 것입니다. 숙련된 실무자와 초보자를 포함한 데이터 과학자, 소프트웨어 엔지니어 및 제품 관리자는 실제 ML 애플리케이션을 단계별로 구축하는 데 관련된 도구, 모범 사례 및 과제를 배우게 될 것입니다.
