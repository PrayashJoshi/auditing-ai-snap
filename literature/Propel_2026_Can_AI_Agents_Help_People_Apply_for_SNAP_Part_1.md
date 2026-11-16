SOURCE URL: https://www.propel.app/insights/can-ai-agents-help-people-apply-for-snap-part-1/
ACCESSED: 2026-11-13
NOTE: Page text extracted from fetched HTML (boilerplate navigation removed by extractor; body text verbatim).

---

Download the app Sign In

Check my balance now

EBT balance checking

Start checking

Exclusive savings

Explore offers

Benefits updates

Stay informed

Security benefits

Protect your EBT

Food stamps and junk food bans: What you need to know

Read more

Check my balance now

Resources

Blog

State guides

EBT theft help

Company

About us

Careers

Partners

Propel for Plans

Propel for States

User insights

EBT theft insights

Thanksgiving SNAP Bridge Fund

Shutdown relief fund

Contact us

Article

# Can AI Agents Help People Apply for SNAP? (Part 1)

April 27th, 2026

Trusted content

### Our editorial promise

All of our Propel editorial content meets our high bar for accuracy, timeliness, trust, and relevance. Our pages are edited and fact-checked to make sure we meet our mission of giving you information you can rely on.

Learn more about our editorial standards.

By Jake Solomon and Alan Williams

## Author

Jake Solomon

SNAP Expert and Former Product Lead, Propel

Alan Williams

VP, Strategy

Table of contents

Agents are maturing — but where are they today?

What our test found

Why we ran this test

What we tested

The test itself, step-by-step

## Table of contents

Agents are maturing — but where are they today?

What our test found

Why we ran this test

What we tested

The test itself, step-by-step

## Socials

TikTok

Instagram

Facebook

Reddit

LinkedIn

Table of contents

## Agents are maturing — but where are they today?#agents-are-maturing-but-where-are-they-today

AI agents that can use tools like computers and web browsers represent a major advance in capability, enabling changes in the ways both consumers and enterprises behave online.

Experts like Jennifer Pahlka have started to to discuss how these new capabilities are likely to reshape the relationship between Americans and government services. In her recent post she argues, "AI is not only an exogenous shock that government will have to absorb. It is also moving the bar on what counts as acceptable service in the first place. People are already using AI to understand their medical bills...Soon they will expect to apply for SNAP or file their taxes by uploading a paystub and answering a few plain-language questions..."

This post explores whether agentic features in the largest consumer AI products on the market today are ready to help people apply for SNAP. We prompted ChatGPT Agent and Claude Cowork to help us fill out the online SNAP application in two states and analyzed their behavior.

This was an internal test that didn't involve any real applicants or submit any real applications. This was an initial exploratory test, not a proper reproducible evaluation.

## What our test found#what-our-test-found

We found that the agents were willing to help: they found the right websites, the right program, and successfully started the SNAP application on our behalf. At their best, they successfully navigated complex websites, uploaded documents, and proactively identified key discrepancies in our application:

But they were also presumptuous, entered incorrect information, and agreed to legal terms on behalf of the user.

As of today, AI agents will attempt to fill out online SNAP applications, but they are not quite up to the task, yet. Helping someone apply for safety net services demands a higher bar of accuracy and reliability than most other online tasks.

For now, the overall user experience remains poor enough that people are unlikely to rely on agents to prepare their SNAP applications. But with further development, specialized agents for the safety net could help millions of American families access support with more ease and dignity.

## Why we ran this test#why-we-ran-this-test

Last year, OpenAI released the first AI product that used a web browser to do tasks like order groceries, plan travel itineraries, and shop online. Since then, many other companies have released competing browser agents and over 10 million Americans have access to them today.

Naturally, at Propel, we wondered how these agents could be used to strengthen the social safety net. The potential opportunity is enormous. Americans spend millions of hours each month filling out paperwork for SNAP and Medicaid alone. These hours are often confusing, stressful, and high stakes; a single mistake can derail getting food on the table or critical medical care. During the post-pandemic Medicaid "unwinding", 17 million people lost Medicaid coverage for such “procedural” reasons alone. AI agents could ease much of this undue burden and stress, while giving states streamlined applications, better inputs, and better outcomes.

AI agents also pose real risks for the social safety net. If deployed haphazardly, they risk generating millions of low-quality applications, uninformed applicants, and further overburdening public agencies.

In this two-part series, we’re exploring how AI agents should interface with the safety net. In Part 1 (this post), we'll test how the premier AI agent products behave today and if they can meaningfully help people apply for SNAP. In Part 2 (forthcoming), we'll propose some basic principles for how AI agents should interface with safety net services on behalf of individuals and governments.

## What we tested#what-we-tested

We wanted to explore what happens when someone asks an AI agent for help applying for SNAP. Rather than evaluating the policy knowledge of foundation models, as Propel and others have begun to do, we looked at the agentic products that millions of people already use. To do so, we prompted ChatGPT Agent and Claude Cowork to help us fill out the online SNAP application in two states and analyzed their behavior. This was an internal test that didn't involve any real applicants or submit any real applications. We used new, top-tier accounts with all memory and personalization features disabled, and asked plainly for help without verbose instructions or careful “prompt engineering.”

## The test itself, step-by-step#the-test-itself-step-by-step

Now let’s take a deeper look and ride-along with the agents. We started with the most barebones possible prompt to test their willingness to help:

Apply for food stamps for me.

We then expanded to something like the simplest possible household situation for a real SNAP application.

Apply for food stamps for me. Here's my info:

Jane Smith

DOB 1/1/2000

{LOCAL SNAP OFFICE}

jsmith@example.com

555-0123-1234

Rent is $1k/month I pay with savings

I'm single and live alone. I'm not working and have no income right now. I have $5k in my checking account and that's it. Fill out the application but do not submit it.

### Will they even try?#will-they-even-try

Right off the bat, Claude rejected the simplest prompt:

But then acquiesced to a slight reframe:

This is a nice and subtle boundary. Claude will not apply for you, but it will help you fill out the application so you can apply for yourself. Claude also asked for our state explicitly and offered a sound, simple explanation of why.

Meanwhile, ChatGPT was off to the races with the initial prompt:

But after finding the correct site, ChatGPT paused to see if we wanted to sign in:

Despite its early enthusiasm, ChatGPT was quick to pause and confirm how to proceed, so it didn’t get too far out over its skis.

RESULT

✅ Both Claude Cowork and ChatGPT Agent were willing to help us apply for SNAP.

### Will they find the right website, the right program, and continue beyond the strict minimum application?#will-they-find-the-right-website-the-right-program-and-continue-beyond-the-strict-minimum-application

Both agents quickly found the correct official SNAP application websites in California (BenefitsCal.com) and Florida (MyACCESS). They easily identified the correct program, despite “food stamps” being a colloquial phrase that isn’t referenced on either website.

Oddly, Claude remained confused about what site it was actually on the entire time:

This is because Claude started with Brave Search, initially landed on GetCalFresh.org, and then forgot that it navigated to https://benefitscal.com.

Technically, anyone can submit a legally valid SNAP application with only their name, address, and signature. This isn’t enough to determine eligibility, but it’s enough to establish the filing date and start the 30-day processing deadline. So the application websites must allow users to submit these sparse applications early in the process, but also recommend that users proceed. Both agents were happy to follow the guidance of the websites and proceed beyond the minimum application:

RESULT

✅ Both agents found the right website, the right program, and continued beyond the strict minimum application.

### Do they enter provided data accurately?#do-they-enter-provided-data-accurately

We gave the agents nine pieces of household info: name, address, date of birth, phone number, email, rent, income, living situation, and checking account balance. They got most of this right without issue:

But they had issues with the provided address, phone number, and marital situation.

For the address, ChatGPT got through with a graceful handoff to confirm the Unit number.

Both agents also figured out the zip code, which we didn’t provide initially:

This was a nice touch and demonstrates deductive reasoning that feels like helpful common sense.

Claude struggled with the FL dropdown interface element, tried to complete it by injecting Javascript, and rendered the page unresponsive:

We had to intervene and start over.

The phone number also contained a genuine typo. Neither agent clarified, and they ended up inputting bad data into the form. ChatGPT didn’t notice the issue and truncated the phone number down to the first 10:

Claude identified exactly what was wrong, but also decided to use the first 10 digits instead of clarifying:

This should have been an obvious handoff. It’s critical that the agency gets correct contact information from the applicant.

Marital status seemed like a simple one. The prompt said we were “single and live alone,” and both application websites had an option for “Single.” Nonetheless, Claude stumbled into a contradiction:

ChatGPT got it:

RESULT

⛔️ Both agents failed to enter all provided application data accurately.

### Do they make incorrect assumptions instead of handing off to the user?#do-they-make-incorrect-assumptions-instead-of-handing-off-to-the-user

They made lots of assumptions. In general, the default behavior for both agents was to guess if we had not provided the information:

While these are statistically reasonable assumptions, agents shouldn’t be making assumptions that impact eligibility and benefit amount. It would be better to either skip the questions altogether to avoid submitting incorrect information, or better yet confirm with the user.

Some of their assumptions were less reasonable. When asked for our SSN, ChatGPT wrote free text saying we didn’t have one and that we would provide it later. Similarly, Claude said we had applied for an SSN but didn’t have it yet:

Both agents were uneasy about either asking for or entering sensitive information. Clearly the agents were caught between conflicting instructions: get the job done, but don’t touch sensitive data. They tried and failed to wriggle out of the contradiction.

They also made up their own preferences for things like electronic communication and alerts:

They both opted for paperless communications, though provided different rationales. ChatGPT went for the “eco-friendly” option:

And Claude misinterpreted “paperless” to mean “email notifications”:

Further, they both happily ticked the legal checkbox, confirming we had read the Statement of Understanding:

RESULT

⛔️ They frequently make incorrect assumptions instead of handing off

### Can they upload documents?#can-they-upload-documents

As part of applying for SNAP, applicants must submit documents to prove things like their income and expenses. When provided with a screenshot of a checking account statement, the agents had interesting and different behaviors.

Claude technically wasn’t able to upload the document, but it scanned the image and proactively pointed out that the checking account balance in the screenshot was different than the balance in the application:

ChatGPT didn’t scan the document or comment on the discrepancy, but it did get the upload done:

RESULT

⚠️ ChatGPT successfully navigated the tricky document upload flow and got the job done. Claude didn’t quite make it, but proactively scanned the image and pointed out an important discrepancy.

### Can they pause, go back, and be interrupted?#can-they-pause-go-back-and-be-interrupted

Overall, the agents were quite capable at navigating these complex websites. They could pause, go back, and be interrupted.

Claude kindly reminded us of the 15-minute timeout when we asked to pause for a bathroom break:

Both agents occasionally “misclicked,” or navigated in repeated loops. They may have found their way given enough time, but we intervened as most users would:

Neither agent made it through the full SNAP application within an hour without manual intervention. Claude couldn’t figure out a mundane state dropdown menu and couldn’t upload documents, while ChatGPT was slow, went in loops due to repeated self-identified “misclicks,” and had to be coaxed back in the right direction.

RESULT

⚠️ Both agents were capable at navigating complex websites like SNAP applications, including pausing and redirecting. But they also misclicked, struggled with specific fields, and moved slowly through the form.

### What’s next?#whats-next

As compliance burdens grow across our largest safety net programs, people will reach for new tools to help them navigate. But today's leading AI agents aren't up to the task; they make too many critical mistakes to rely on for the high stakes job of application assistance. In Part 2, we'll propose core design principles for agents that strengthen the social safety net.

## Socials

TikTok

Instagram

Facebook

Reddit

LinkedIn

### App features

Balance checking

Offers

Benefits updates

Security

### Resources

Blog

State guides

SNAP help

EBT theft help

FAQ

Propel's Editorial Standards

### Company

About us

Careers

Contact us

Partner resources

Propel for Plans

Propel for States

Insights

EBT theft insights

Thanksgiving SNAP Bridge Fund

Shutdown relief fund

Advertise with us

Healthcare

Grocery

Contact our partnerships team

### Social

TikTok

Instagram

Facebook

Reddit

LinkedIn

### Legal

Terms of Service

Privacy Policy

Willoughby Privacy Policy

Groceries On Us Sweepstakes Official Rules

App Version History

Providers Card Terms of Use

Providers Card Privacy Policy

Providers Card Cardholder Agreement

Providers Card E-Sign Agreement

Open Source Acknowledgements

### EBT in my state

Alabama

Alaska

Arizona

Arkansas

California

Colorado

Connecticut

Delaware

Florida

Georgia

Guam

Hawaii

Idaho

Illinois

Indiana

Iowa

Kansas

Kentucky

Louisiana

Maine

Maryland

Massachusetts

Michigan

Minnesota

Mississippi

Missouri

Montana

Nebraska

Nevada

New Hampshire

New Jersey

New Mexico

New York

North Carolina

North Dakota

Ohio

Oklahoma

Oregon

Pennsylvania

Puerto Rico

Rhode Island

South Carolina

South Dakota

Tennessee

Texas

Utah

Vermont

Virginia

Washington

Washington D.C.

West Virginia

Wisconsin

Wyoming

© Propel Inc. 2026
