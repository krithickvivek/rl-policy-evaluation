# POLICY EVALUATION

## AIM
To evaluate and compare the effectiveness of two different policies in the FrozenLake environment by calculating their state-value functions using the policy evaluation algorithm. This will help determine which policy yields better expected returns and performance in navigating the environment.

## PROBLEM STATEMENT
The FrozenLake environment is a grid-world where an agent must navigate from a start state to a goal state while avoiding holes. Given two predefined policies, the problem is to evaluate their performance by computing their state-value functions using policy evaluation. This helps to understand which policy leads to better expected rewards and safer navigation under the environment’s stochastic dynamics.

## POLICY EVALUATION FUNCTION
``` python
# Name: Krithick Vivekananda
# Reg.No: 212223240075
import numpy as np
def policy_evaluation(pi,P,gamma=1.0,theta=1e-10):
  prev_V=np.zeros(len(P))
  while True:
    V=np.zeros(len(P))
    for s in range(len(P)):
      for prob, next_state, reward, done in P[s][pi(s)]:
        V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
    if np.max(np.abs(prev_V-V)) < theta:
      break
    prev_V=V.copy()
  return V
```

## OUTPUT:
<img width="509" height="125" alt="image" src="https://github.com/user-attachments/assets/21419329-72b6-4961-bcd8-9686075970f5" />
<img width="772" height="61" alt="image" src="https://github.com/user-attachments/assets/9aeced03-009f-47ee-b294-1896d1dbe410" />
<img width="514" height="178" alt="image" src="https://github.com/user-attachments/assets/14ab1c63-e8b5-4ed9-973c-d2b503000837" />
<img width="706" height="76" alt="image" src="https://github.com/user-attachments/assets/3c626b2f-1cd0-4986-ba1c-6f004ebf9908" />

- Policy → FrozenLake has a 10% success rate, while Policy → 2 has 5%.
- Policy → FrozenLake has a 0.1000 average return, vs. 0.0500 for Policy → 2.
- Policy → FrozenLake uses varied directions; Policy → 2 is mostly RIGHT/DOWN.
- Policy → FrozenLake is more adaptable and avoids holes better.
- ✅ Policy → FrozenLake performs better overall.
  
<img width="575" height="152" alt="image" src="https://github.com/user-attachments/assets/c1918d75-db33-4e47-8c41-71a77fec80bb" />
<img width="618" height="149" alt="image" src="https://github.com/user-attachments/assets/31e5cf3e-2899-44ce-8c3e-cdfd72d2b11a" />

- Overall values are higher for pi_frozenlake than for pi_2.
- pi_frozenlake has high values in key states like 6 (0.20562), 8 (0.30562), 14 (0.80739).
- pi_2 shows lower values in the same states: 6 (0.09215), 8 (0.1196), 14 (0.75408).
- pi_frozenlake maintains better value propagation across the grid.
- ✅ pi_frozenlake has a better state-value function, indicating a stronger policy.
  
<img width="594" height="213" alt="image" src="https://github.com/user-attachments/assets/2d4e6e07-83ee-4998-b78d-a4ebf92af147" />

## RESULT:

Therefore, pi_frozenlake is a stronger policy with higher state values, leading to better performance than pi_2.

