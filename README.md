# cv-crowdcrush-detection
Implementation of CSRNet Single-Column CNN Crowd Counting model + notification of crowd crush danger if exceeding certain threshold. 

# design decisions
I chose to use a single-column CNN instead of multi-column CNN as it would be very computationally expensive to run and optimize a multi-column model. Specifically, CSRNet model was chosen as it has the most documentation of their architecture and parameters.
