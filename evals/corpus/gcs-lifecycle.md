# Cloud Storage lifecycle and storage classes

A lifecycle rule acts on objects once a condition is met. The most common
condition is age: delete an object a fixed number of days after creation. Rules
run asynchronously, so deletion happens within roughly a day of the condition
becoming true rather than at the exact moment.

Storage classes trade retrieval cost against storage cost. Standard suits data
read regularly. Nearline, Coldline and Archive each lower the storage price and
raise both the retrieval price and the minimum storage duration. Deleting an
Archive object before ninety days still incurs the full ninety-day charge.

Object versioning keeps previous generations of an overwritten object. It is
off by default, and turning it on without a lifecycle rule to expire old
generations is a common way to exceed a free allowance without noticing.

Uniform bucket-level access disables per-object ACLs and makes IAM the only
access mechanism. It is the recommended setting because per-object ACLs are
difficult to audit.
