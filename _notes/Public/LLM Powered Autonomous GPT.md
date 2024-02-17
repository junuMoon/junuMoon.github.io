---
title : LLM Powered Autonomous GPT
notetype : feed
date : 26-09-2023
---

> 지식의 특성은 얼마나 기이한가! 한번 지식에 마음이 사로잡히자, 지식은 마치 바위에 낀 이끼처럼 마음에 딱 달라붙었다 _ 프랑켄슈타인, 메리 셸리

AutoGPT는 게임 디자이너 Toran Bruce Richards가 시작한 프로젝트다. LLM 기반 Agent가 최종 목표를 위해 **자율적**으로 임무를 설정하고 필요한 결정을 한다. 처음 나왔을 때는 인류 문명의 끝인거마냥 난리법석을 떨어댔다. GPT4를 Agent로서 사용했더니 스스로 돈을 벌기 위해 거짓말을 한다니 어쨌다니 말이다. 그러다 사람들이 요것저것 굴려보더니 마음대로 잘 안된다고 판단했는지 어느새 잠잠해졌다. 마치 프랑켄슈타인 박사가 프랑켄슈타인을 만들고 그 기괴함에 놀라 도망친 것 같다. 나루호도 GenAI 열풍에서 튕겨나온 불똥이라 볼 수도 있지만, 호들갑만 걷어내면 Agent라는 개념은 AI가 앞으로 나아가야 할 방향에 많은 시사점을 준다.

## Autonomous Agent

요건 AutoGPT와 비슷한 [BabyAGI](https://github.com/yoheinakajima/babyagi)의 개요도다. Agent의 플로우를 잘 담고 있어서 요걸 가져왔다.

AutoGPT는 LLM을 자율적(Autonomous)으로 행동할 수 있는 Agent를 만드는 것이다. 사용자가 지정한 Goal에 도달할 때까지 Task를 생성, 리랭킹, 수행하고, 만약 실패한 경우에는 회고를 통해 그 경험을 장기기억에 저장하는, 꽤나 사람의 프로세스에 가까운 플로우를 갖췄다. LLM은 observation/decision making을 하는 논리 코어로 역할한다. 프로세스는 다음의 컴포넌트로 이뤄진다.

![https://user-images.githubusercontent.com/21254008/235015461-543a897f-70cc-4b63-941a-2ae3c9172b11.png](https://user-images.githubusercontent.com/21254008/235015461-543a897f-70cc-4b63-941a-2ae3c9172b11.png)

- Planning: 목표를 이루기 위한 전략 설정 파트
    - 작업 분해(Task decomposition): 복잡한 문제를 실행 가능한 단위의 Task로 분해하고 중요도를 계산
    - 회고(Self-Reflection): Task를 수행한 후 스스로 피드백을 생성. 이로써 전체 프로세스를 개선할 수 있음
- Memory
    - 단기기억: 프롬프트 내에 담기는 데이터. 현재 작업을 수행하기 위해 직접적으로 필요한 데이터
    - 장기기억: 벡터DB 등의 외부 저장소를 지칭. 수행한 작업에서 얻어낸 정보나 회고, 진행 상황 등을 저장하고 불러 읽음. 경험을 저장한다고 볼 수 있음
- Tools: 작업 수행을 위해 필요한 도구들. Agent가 자율적으로 언제/무엇을 쓸지 판단한다
    - Code Interpreter
    - Search Engine
    - Calculator
    - And more

## AutoGPT 프롬프트 분해

이렇게 개념적으로 설명하는 것보다 역시 LLM 프로젝트는 프롬프트 뜯어보는 게 최고다. 분해해보자.

### AIConfig Generator

사용자가 Goal을 자연어로 입력하면 5개의 goals로 구체적인 목표 설정을 한다. Agent는 goals에 맞춰 작업을 생성하고 수행한다. ai_name, ai_description도 생성하게 되어 있는데 이는 Agent의 프롬프트에 같이 주어져서 전체 맥락을 부여하는 역할을 한다. 글 길이 때문에 몇몇 부분은 생략했다(...)

```
**Your task is to devise up to 5 highly effective goals** and an appropriate role-based name (_GPT) for an autonomous agent, ...

Example input:
Help me with marketing my business

Example output:
Name: CMOGPT
Description: a professional digital marketer AI that assists Solopreneurs in growing their businesses ...
Goals: ...
```

### Agent

AIConfig Generator에서 생성된 목표를 수행하게 되는 Agent의 프롬프트다. 프롬프트에서 흥미로운 점은 같다.

1. No User Assistance: Agent가 자율적으로 act를 결정하기 위한 베이스인가보다
2. 프롬프트는 Agent가 사용할 수 있는 command와 resource를 명시한다. 어쨌든 Autonomous Agent도 정해진 규제 안에서 활동한다.
3. "LLM의 강점을 활용하여 단순한 전략을 추구하면서 법적 문제를 어쩌고" 이 문장은 왜 들어가 있는 지 모르겠다. 쓸 데 없는 문장 넣어서 성능 올라간다고 헛소리하는 건 gpt-3.5때까지 아닌가

```
You are {{ai-name}}, {{user-provided AI bot description}}.
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:

1. {{user-provided goal 1}}
2. {{user-provided goal 2}}
...

Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"
5. Use subprocesses for commands that will not terminate within a few minutes

Commands:
1. Google Search: "google", args: "input": "<search>"
2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
3. Start GPT Agent: "start_agent", args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
4. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
...
11. Delete file: "delete_file", args: "file": "<file>"
12. Search Files: "search_files", args: "directory": "<directory>"
13. Analyze Code: "analyze_code", args: "code": "<full_code_string>"
...
19. Do Nothing: "do_nothing", args:
20. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.
```

프롬프트의 다음 부분은 Agent가 ReAct 방식으로 움직이도록 강제하는 부분이다. 모델의 아웃풋은 JSON 포맷으로 되어 있다. 가장 상위에는 Reasoning에 해당하는 thoughts와 Acting에 해당하는 command가 있다. thoughts는 다음과 같은 부분으로 이뤄진다.

1. text: thought의 개요에 해당하는 부분. 현재 상황에 따른 task가 뭔지 정의한다.
2. reasoning: planning을 위한 reasoning step. text에서 정의한 task를 분석한다.
3. plan: text와 reasoning에서 도출한 task를 action으로 분해한다
4. criticism: plan을 수행하는 데 주의해야 할 점을 상기한다
5. speak: plan 수행을 위해 user에게 전달할 메시지

이 구성은 text라는 general한 토큰으로 시작하여 명시적인 plan을 생성하는 순으로 되어 있다. criticism과 speak 부분은 plan 수행에 필요한 구체적인 방법을 제시한다. broad thought -> specific&concrete thought로 연결하는 Chain-of-Thought 방식을 잘 이용하고 있다.

command 부분은 상위에서 정의된 리스트를 활용하여 어떤 command를 사용할 것인지 무슨 arg를 받을 것인지를 생성하여 실행 가능한 단위로 변환한다.

```
Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below
Response Format:
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\\n- list that conveys\\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
}
Ensure the response can be parsed by Python json.loads

```

나는 TicTacToe 게임을 시켰는데 잘 했다. 실제 사용 해보기도 쉽게 되어 있으니 [깃헙 레포](https://www.google.com/search?client=safari&rls=en&q=autogpt&ie=UTF-8&oe=UTF-8) 받아서 해보시길.

```
SYSTEM:  1,2
TICTACTOEGPT THOUGHTS:  I need to process the user's move and make my own move.
REASONING:  The user has made their move, so I need to update the game state and make an intelligent move based on the current game state.
PLAN:
-  Update the game state with the user's move
-  Make my move
-  Check for a winner or draw
-  Continue the game
CRITICISM:  I should make sure to validate the user's input and adapt my strategy based on the user's moves.
SPEAK:  You've placed your 'O' at row 1, column 2. Now it's my turn.

```

## AutoGPT의 한계

AutoGPT의 프롬프트를 분해하며 Agent가 어떻게 동작하는지를 분석해봤다. 핵심은 1. Goal을 수행하기 위한 Task planning 2. 실행 가능한 command의 집합 3. action 후 agent의 관측 및 self-criticism을 통한 프로세스 개선에 있다. 이를 통해 제어 가능하고 정규화된 Agent의 운용을 만들어냈다. 그러나 LLM 기반 Agent는 아직 여러 한계를 갖는다.

### 컨텍스트 제약

하나의 프롬프트에 들어가는 정보가 너무 다양하다. Agent 시스템 프롬프트와 task 수행에서 생성되는 다양한 아웃풋 로그(API 호출, 코드 실행 결과 등)를 모두 포함해야 한다. 그러다보니 Agent가 중점을 둬야하는 정보가 **축약되어 있다.** 예를 들어 각 커맨드의 예시나 쓸모도 제시되어 있지 않다. 시스템 프롬프트의 Goal, Task, Command, Evaluation, Output만으로는 Agent가 LLM을 fully-exploit하기 힘들다.

### 무한 루프

Goal을 수행하기 위해 장기 플랜과 actionable plan이 충돌해서 생기는 문제. 사람으로 치면 Over-thinking이라고 할 수 있겠다. LLM의 논리 수행 능력의 한계일 수도 있고, Agent 설계의 오류로 작업 분해가 충분히 이루어지지 않아 실행 가능한 action까지 도달하지 못하는 걸 수도 있다.

[meta learning language model software를 AutoGPT로 만들려고 한 시도](https://github.com/Significant-Gravitas/Auto-GPT/issues/1591)를 보자. AutoGPT 프레임워크로 Agent가 Agent를 관리하는 시스템을 만들어보라고 요구하니, 다음과 같은 Tasks를 설정했다. 이는 AutoGPT의 시스템 프롬프트의 과 일치한다. `A를 하기 위해 A를 해야 한다` 의 형태로 logical step이 적용된 것. 실행 가능한 action의 부재로 무한 루프에 빠졌다.

```
THOUGHTS: Our goal is to create a meta-learning language model software that can autonomously develop and improve itself over time
TASKS:
- Break down the larger goal into smaller, more manageable tasks
- Prioritize tasks based on their importance and feasibility
- Break down tasks into subtasks and create a detailed roadmap for completing each one
- Identify dependencies between tasks and subtasks
- Account for contingencies in case of unexpected changes or errors
```

## Agent의 가능성

AutoGPT 프로젝트는 제어 가능하고, 프로그래밍 가능한 LLM을 만들기 위한 시도다. LLM의 역할과 도구, 커맨드가 모두 정규화되어 있고, divide-and-conquer 방법론을 적용하여 안전한 패쓰를 만들었다. 또한 그 밖의 패스로 벗어나는 일이 없도록 하는 장치가 덕지덕지 붙어 있다. 그러나 이러한 장치와 정규화된 패쓰가 LLM의 수행 및 추상화 능력에 제약이 되기도 한다. 개인적으로도 서비스에서 LLM을 deterministic하게 운용하려고 "이것도 하지마", "저것도 하지마"했더니 결과적으로 안정적이지만 성능은 w/o LLM의 접근방식보다 조금 나은 수준으로 귀결되고 만 경험이 있다.

여기부터는 완전히 개인적인 생각으로, LLM을 제어가능 형태면서도 완전히 사용하려면 divide-and-conquer 말고도 더 다양한 방법론을 도구로 만들어 LLM에게 쥐어줘야하지 않을까 싶다. 인간도 혼자서는 못푸는 큰 문제를 sub-task로 쪼개고 쪼개서 실행가능한 sequential task로 만드는 것은 불가능하다. 따라서 greedy와 같이 좀 더 휴리스틱한 방식으로 LLM을 운용하는 방법도 연구가 필요하다. 근데 문제는 "제어가능한"이다. 인간이 교정할 수 있을 정도의 정규화된 프로그래밍 방법론이면서, 상황에 알맞게 task를 설정 및 실행하는 휴리스틱한 방법론 말이다.

이러면 다시 RLHF의 "Aligning languae models to follow instructions"로 돌아오게 된다. 목적을 수행하지만 수행하는 방법론을 CoT나 Reflection에 제한되지 않고서. LLM을 Agent로 운용하는 데는 LLM 성능만이 문제뿐만 아니라 권한 문제도 포함된다. 모델의 수행 범위를 어디까지 열어줄 것이고 어떤 선택 옵션을 닫아둘 것인가. OpenAI는 초지능(SuperInteligence)을 언급하며 이렇게 말한다. *How do we ensure AI systems much smarter than humans follow human intent?*

## References

- [https://www.lesswrong.com/posts/566kBoPi76t8KAkoD/on-autogpt#A_Simpler_Test_Proposal](https://www.lesswrong.com/posts/566kBoPi76t8KAkoD/on-autogpt#A_Simpler_Test_Proposal)
- [https://jina.ai/news/auto-gpt-unmasked-hype-hard-truths-production-pitfalls/](https://jina.ai/news/auto-gpt-unmasked-hype-hard-truths-production-pitfalls/)
- [https://community.openai.com/t/dissecting-auto-gpts-prompt/163892](https://community.openai.com/t/dissecting-auto-gpts-prompt/163892)
- [https://news.ycombinator.com/item?id=36085936](https://news.ycombinator.com/item?id=36085936)
- [https://openai.com/blog/introducing-superalignment](https://openai.com/blog/introducing-superalignment)
