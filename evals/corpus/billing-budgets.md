# Billing budgets and alerts

A budget notifies; it does not cap. Resources continue running past the budget
amount, and the only automated way to stop spending is to act on the
notification, for example by triggering a function that disables billing.

Threshold rules fire at percentages of the budget. Rules can act on actual
spend or on forecast spend, and a forecast rule warns before the money is
gone rather than after.

Credit treatment determines what the budget counts. Including credits tracks
gross usage, so a budget fires while trial credits are being consumed.
Excluding them tracks only out-of-pocket spend, which stays at zero until the
credits are exhausted -- by which point the drawdown has already happened.

Budgets are scoped to a billing account and optionally filtered to projects,
services or labels. A budget with no filter covers every project on the
account.
