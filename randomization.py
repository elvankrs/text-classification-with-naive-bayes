import random
from evaluate import evaluate

def shuffle(a, b, prob):

  if random.random() < prob:
    a, b = b, a

  return a, b

def compute_s(targets, preds_A, preds_B):
  F_A, F_B = evaluate(targets, preds_A, verbose=False), evaluate(targets, preds_B, verbose=False) # Macro-F scores of system A & B
  s = abs(F_A - F_B)
  return s

def randomization_test(targets_multinomial, preds_multinomial, preds_bernoulli):
   R = 1000
   counter = 0
   rejection_level = 0.05

   s = compute_s(targets_multinomial, preds_multinomial, preds_bernoulli) # Compute test statistics s(A,B) = |F(A) - F(B)| on the test data. (F=macro-avg)

   for i in range(R):
      # if i % 100 == 0: print(i)
      targets_new, preds_A_new, preds_B_new = [], [], []
      for target, pred_A, pred_B in zip(targets_multinomial, preds_multinomial, preds_bernoulli): # Get the outputs of system A and system B
         pred_A_new, pred_B_new = shuffle(pred_A, pred_B, 0.5) # Shuffle a and b with probability 0.5
         preds_A_new.append(pred_A_new)
         preds_B_new.append(pred_B_new)
         targets_new.append(target)

      s_new = compute_s(targets_new, preds_A_new, preds_B_new) # Compute pseudo-statistics s*(A', B')  = |F(A') - F(B')| on shuffled data
      if s_new > s:
         counter += 1

   p = (counter + 1) / (R + 1)
   if p <= rejection_level:
      print(f"p: {p:.4f} <= {rejection_level}, hence we reject the null hyptohesis.")
   else:
      print(f"p: {p:.4f} > {rejection_level}, hence there is not sufficient evidence to reject the null hyptohesis.")
