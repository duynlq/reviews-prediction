/* Unique Hotel Names */
select distinct date_stayed
from dbo.reviews
order by date_stayed

/* Unique Date Stayed */
select distinct date_stayed
from dbo.reviews
order by date_stayed

/* view oldest review */
select *
from dbo.reviews
where date_stayed='December 2007'

/* find percentage of 2023 review count vs total review count */
select (COUNT(case when date_stayed LIKE '%2023%' then 1 end) * 100.0) / COUNT(*) as '2023_percentage'
from dbo.reviews;

/* add and update table with a new column "churn" */
alter table dbo.reviews add churn text;
UPDATE dbo.reviews 
set churn = case when processed_rating < 40 then 'churn' else 'non-churn' end;

/* overall view */
select * from dbo.reviews