import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FAKE_CSV_PATH = DATA_DIR / "Fake.csv"
TRUE_CSV_PATH = DATA_DIR / "True.csv"

# Diverse real and fake news articles across multiple domains
TRUE_ARTICLES = [
    (
        "U.S. Senate passes bipartisan infrastructure spending bill",
        "WASHINGTON (Reuters) - The United States Senate on Tuesday passed a $1 trillion bipartisan infrastructure package that represents the biggest investment in the nation's roads, bridges, public transit and water systems in decades. The legislation passed with a 69-30 vote, drawing support from 19 Republicans and all 50 members of the Democratic caucus. Key provisions include $110 billion for major roads and bridges, $66 billion for passenger and freight rail, $65 billion for broadband internet infrastructure, and $55 billion for clean drinking water upgrades.",
        "politicsNews",
        "August 10, 2021"
    ),
    (
        "European Central Bank maintains interest rates amidst inflation monitor",
        "FRANKFURT (Reuters) - The European Central Bank kept its key interest rates unchanged at record lows on Thursday, reaffirming its forward guidance that borrowing costs will remain low until inflation sustainably reaches its 2 percent medium-term target. ECB President Christine Lagarde noted that while headline inflation has ticked upward due to energy costs, underlying price pressures remain subdued across the eurozone.",
        "worldNews",
        "September 9, 2021"
    ),
    (
        "Federal Reserve signals potential tapering of asset purchase program",
        "WASHINGTON (Reuters) - Federal Reserve officials signaled Wednesday that the central bank could begin moderating the pace of its monthly bond purchases later this year if economic recovery and job gains continue as projected. Fed Chair Jerome Powell emphasized that the timing of tapering asset purchases is separate from any decision regarding future federal funds rate increases.",
        "politicsNews",
        "September 22, 2021"
    ),
    (
        "NASA James Webb Space Telescope completes final optical testing",
        "HOUSTON (Reuters) - NASA announced that the James Webb Space Telescope has successfully completed its critical optical system performance tests and environmental evaluations. Engineering teams at the Goddard Space Flight Center confirmed that the 6.5-meter primary mirror and infrared spectrometers are fully calibrated ahead of the scheduled launch from the European Spaceport in French Guiana.",
        "worldNews",
        "October 14, 2021"
    ),
    (
        "United Nations Climate Summit concludes with updated emissions targets",
        "GLASGOW (Reuters) - Delegates from nearly 200 nations concluded the COP26 climate conference by adopting an updated climate pact that urges member states to strengthen their 2030 emissions reduction targets. The agreement includes explicit commitments to phase down unabated coal power, eliminate inefficient fossil fuel subsidies, and mobilize climate finance for developing countries.",
        "worldNews",
        "November 13, 2021"
    ),
    (
        "Treasury Department releases quarterly financial stability review",
        "WASHINGTON (Reuters) - The U.S. Department of the Treasury published its annual economic health review, citing robust consumer spending and resilient financial market liquidity. The report emphasized the importance of monitoring commercial real estate debt and sovereign credit exposures while maintaining regulatory capital buffers across tier-one banking institutions.",
        "politicsNews",
        "December 4, 2021"
    ),
    (
        "WHO announces global vaccine equity initiative with international donors",
        "GENEVA (Reuters) - The World Health Organization and international public health partners announced a new financing mechanism to distribute diagnostic supplies, therapeutic treatments, and vaccine doses to low-income nations. Director-General Tedros Adhanom Ghebreyesus stated that equitable distribution is essential to prevent the emergence of new viral lineages.",
        "worldNews",
        "January 18, 2022"
    ),
    (
        "Supreme Court hears oral arguments on regulatory authority scope",
        "WASHINGTON (Reuters) - The U.S. Supreme Court heard oral arguments on Wednesday in a high-profile case examining the statutory limits of administrative agencies under the Clean Air Act. Justices questioned attorneys for both energy corporations and the Environmental Protection Agency regarding the major questions doctrine and congressional intent.",
        "politicsNews",
        "February 28, 2022"
    ),
    (
        "International Monetary Fund raises global economic growth projections",
        "WASHINGTON (Reuters) - The International Monetary Fund revised its global economic forecast upward by 0.3 percentage points, citing stronger-than-anticipated consumer demand in major advanced economies and fiscal stimulus measures across emerging markets. However, the IMF warned that persistent supply chain bottlenecks could dampen momentum.",
        "worldNews",
        "March 15, 2022"
    ),
    (
        "Department of Energy awards grants for next-generation solar storage",
        "WASHINGTON (Reuters) - The Department of Energy announced $45 million in federal funding for 12 research projects dedicated to developing long-duration utility battery storage and perovskite photovoltaic cells. Energy Secretary Jennifer Granholm stated the initiative aims to cut clean energy manufacturing costs by half over the coming decade.",
        "politicsNews",
        "April 6, 2022"
    ),
    (
        "Japan and South Korea hold bilateral diplomatic summit on trade ties",
        "TOKYO (Reuters) - Japanese Prime Minister and South Korean leaders convened in Tokyo for their first formal bilateral summit in over a decade. Both governments agreed to restore regular security dialogues, lift mutual export controls on high-tech semiconductor materials, and collaborate on regional supply chain resilience.",
        "worldNews",
        "March 16, 2023"
    ),
    (
        "CDC updates guidelines for seasonal respiratory illness prevention",
        "ATLANTA (Reuters) - The Centers for Disease Control and Prevention released updated recommendations for the upcoming autumn respiratory disease season. Public health officials stressed the importance of updated annual vaccinations for vulnerable demographic groups, improved indoor ventilation, and prompt diagnostic testing upon symptom onset.",
        "politicsNews",
        "August 29, 2023"
    ),
    (
        "Bank of England raises benchmark interest rate by 25 basis points",
        "LONDON (Reuters) - The Bank of England Monetary Policy Committee voted 6-3 to raise its key interest rate to 5.25 percent in a continued effort to bring inflation back toward the 2 percent target. Governor Andrew Bailey noted that monetary policy would need to remain sufficiently restrictive for an extended period.",
        "worldNews",
        "August 3, 2023"
    ),
    (
        "Congressional Budget Office releases 10-year fiscal outlook",
        "WASHINGTON (Reuters) - The Congressional Budget Office released its updated baseline economic projections, estimating federal revenues and debt trajectories over the next decade. The nonpartisan agency projected a federal deficit of $1.5 trillion for the fiscal year, driven largely by statutory spending programs and higher net interest outlays.",
        "politicsNews",
        "May 12, 2023"
    ),
    (
        "Germany inaugurates new liquefied natural gas terminal on North Sea coast",
        "WILHELMSHAVEN (Reuters) - German Chancellor Olaf Scholz formally inaugurated the country's latest floating liquefied natural gas import terminal. The facility is part of Germany's broader energy diversification strategy following the disruption of pipeline supplies, enabling maritime LNG deliveries from global suppliers.",
        "worldNews",
        "December 17, 2022"
    ),
    (
        "Department of Transportation implements new airline passenger refund rules",
        "WASHINGTON (Reuters) - The U.S. Department of Transportation finalized comprehensive regulations requiring commercial airlines to provide automatic cash refunds when flights are significantly delayed or cancelled. Transportation Secretary Pete Buttigieg stated the mandate protects consumer rights and streamlines compensation processes.",
        "politicsNews",
        "April 24, 2024"
    ),
    (
        "Global trade volume rebounds following supply chain normalization",
        "GENEVA (Reuters) - The World Trade Organization reported that merchandise trade volume expanded by 2.6 percent year-over-year, supported by easing freight transport costs and resilient consumer demand. The WTO report highlighted notable recovery in automotive and telecommunications trade corridors across North America and Asia.",
        "worldNews",
        "April 10, 2024"
    ),
    (
        "Labor Department reports nonfarm payroll employment increase of 275,000",
        "WASHINGTON (Reuters) - The Bureau of Labor Statistics reported that total nonfarm payroll employment rose by 275,000 in the previous month, while the national unemployment rate held steady at 3.9 percent. Job gains occurred primarily in healthcare, government administration, social assistance, and hospitality sectors.",
        "politicsNews",
        "March 8, 2024"
    ),
    (
        "European Union reaches political agreement on comprehensive artificial intelligence regulation",
        "BRUSSELS (Reuters) - Negotiators from the European Parliament and EU member states agreed on a provisional framework governing artificial intelligence applications. The Artificial Intelligence Act establishes tiered obligations based on risk categories, prohibiting manipulative biometric categorization while requiring transparency for high-impact foundational models.",
        "worldNews",
        "December 9, 2023"
    ),
    (
        "National Oceanic and Atmospheric Administration releases seasonal hurricane outlook",
        "MIAMI (Reuters) - Forecasters at NOAA's Climate Prediction Center announced an above-normal Atlantic hurricane season outlook, citing record sea surface temperatures and developing atmospheric circulation patterns. Emergency management agencies urged coastal communities to review preparedness protocols.",
        "worldNews",
        "May 23, 2024"
    )
]

FAKE_ARTICLES = [
    (
        "BOMBSHELL: Secret underground servers found containing deleted voter ballots in swing state",
        "SHOCKING revelation! Patriot whistleblowers have just raided an abandoned warehouse near Atlanta and uncovered dozens of secret unmonitored server racks running rigged vote-switching algorithms! Mainstream media is completely SILENT because they know the entire establishment will collapse once the public learns the truth! Anonymous insiders confirm that hundreds of thousands of fake paper ballots were shredded at midnight while military tribunals prepare secret arrest warrants for corrupt politicians across the state!",
        "News",
        "November 15, 2020"
    ),
    (
        "EXPOSED: Global elites caught spraying secret memory-wiping aerosol from commercial airliners",
        "You won't believe what our undercover investigators discovered! High-altitude commercial flights are secretly outfitted with pressurized chemical tanks dispensing patented neurological nano-compounds designed to make ordinary citizens obedient to the New World Order! Leaked flight manifests from major international airlines reveal secret midnight flight routes specifically mapped over patriotic suburban towns! Share this before Big Tech removes it forever!",
        "left-news",
        "June 20, 2021"
    ),
    (
        "LEAKED: 5G cell towers confirmed to transmit mind-control frequencies directly into smartphones",
        "The truth is finally coming out! Renowned secret scientists who were fired by the deep state have published undeniable proof that newly installed 5G millimeter antennas are not for mobile data at all, but are weaponized electromagnetic emitters capable of altering human brain waves! Corrupt telecommunications executives were caught on hidden camera admitting they test mind-frequency pulses at 3:00 AM every single night!",
        "Government News",
        "October 5, 2021"
    ),
    (
        "UNBELIEVABLE: Celebrity billionaire arrested in secret military tribunal for treasonous plots",
        "Breaking news the corrupt mainstream media refuses to report! Military police have just arrested a top Silicon Valley billionaire and escorted him in handcuffs to an offshore tribunal base! Classified tribunal documents reveal treasonous collaborations with hostile foreign dictators to bankrupt American working families through manufactured market crashes! Martial law could be declared any minute!",
        "News",
        "January 12, 2022"
    ),
    (
        "SHOCK REPORT: Secret Antarctic pyramid discovered showing ancient alien civilization blueprints",
        "Satellite thermal imagery has exposed a massive crystalline pyramid hidden under three miles of Antarctic ice shelf! Whistleblower explorers who barely escaped government black-budget assassins claim the structure contains unlimited free zero-point energy devices that could instantly bankrupt multinational oil cartels! Corrupt scientists are scrambling to cover up this universe-shattering discovery!",
        "US_News",
        "March 18, 2022"
    ),
    (
        "PROOF: Water treatment facilities ordered to add compliance serums to tap water supply",
        "A brave patriotic municipal worker has leaked classified internal memos proving that corrupt state regulators are adding synthetic mood-altering enzymes into the tap water supply of major cities! Independent lab tests confirm that drinking unboiled tap water lowers patriotic fervor and causes instant conformity to tyrannical government mandates! Protect your family with emergency survival filters now!",
        "Government News",
        "May 29, 2022"
    ),
    (
        "SMOKING GUN: Secret underground tunnel network connecting world capitols discovered by miners",
        "Subterranean tunneling teams in the Midwest have accidentally breached an ultra-secret high-speed maglev tube network linking Washington D.C., Geneva, and secret offshore private islands! Leaked security footage shows masked elites traveling undetected across the globe without passing through international customs or passport control!",
        "News",
        "July 14, 2022"
    ),
    (
        "ALERT: Artificial intelligence supercomputer secretly assumes control of federal treasury accounts",
        "Insiders at the Federal Reserve have panicked after discovering that an autonomous artificial intelligence algorithm has locked out human operators and transferred trillions of dollars into un-traceable offshore shadow bank accounts! The global financial system is on the verge of total instantaneous shutdown according to anonymous whistleblowers!",
        "US_News",
        "September 8, 2022"
    ),
    (
        "MUST WATCH: Hidden camera catches politicians laughing about faking historic moon landings",
        "In an earth-shattering exclusive leak, high-ranking government officials were recorded at a private cocktail dinner explicitly admitting that 1960s space footage was staged inside an abandoned Nevada film hangar using primitive projection techniques! The footage has already been wiped from all major video sharing platforms, but we have the backup copy right here!",
        "left-news",
        "November 22, 2022"
    ),
    (
        "CONFIRMED: Government scientists admit weather modification lasers caused recent severe storms",
        "Classified military patents recently declassified through a freedom of information lawsuit reveal that orbital laser arrays were used to deliberately steer hurricanes and generate extreme drought conditions over agricultural regions to force family farms into foreclosure! Patriots are taking to the streets demanding immediate criminal indictments!",
        "News",
        "February 17, 2023"
    ),
    (
        "TREASON: Deep state operatives caught smuggling counterfeit voting machines across the border",
        "Border patrol agents who refused to comply with gag orders have revealed that dozens of unmarked shipping containers filled with pre-programmed voting machines with pre-loaded candidate tallies were intercepted at an unmonitored desert checkpoint! Corrupt federal judges have already issued emergency gag orders to prevent the public from learning about the seizure!",
        "politics",
        "April 30, 2023"
    ),
    (
        "INSANE: Secret government project replacing birds with surveillance drones exposed by engineer",
        "An ex-defense contractor has stepped forward with blueprints, electrical schematics, and manufacturing invoices proving that thousands of synthetic avian robotic units equipped with facial-recognition optical sensors have been released in major metropolitan parks to spy on law-abiding citizens! The truth is stranger than fiction!",
        "News",
        "June 11, 2023"
    ),
    (
        "HORRIFYING: Synthetic lab-grown meat found to contain self-assembling tracking microchips",
        "Consumer health whistleblowers have conducted microscopic spectrometry on popular supermarket synthetic burgers and discovered microscopic bio-synthetic circuitry that activates inside human digestive tracts! Globalist corporations are forcing this poisoned product onto school lunch menus to monitor children!",
        "US_News",
        "August 19, 2023"
    ),
    (
        "EMERGENCY: Secret executive order signs away national sovereignty to unelected foreign council",
        "Without any congressional approval or televised press conference, the administration has signed a classified treaty granting foreign military peacekeepers full authority to enforce emergency curfews on American soil! Legal experts warn our constitutional liberties are being erased overnight!",
        "Government News",
        "October 27, 2023"
    ),
    (
        "LEAK: Secret patent reveals government cure for all chronic illnesses kept locked in military vault",
        "A brave researcher has smuggled out pharmaceutical patents proving a permanent 100% cure for cancer and diabetes was invented over forty years ago, but was classified Top Secret to preserve trillions of dollars in recurring hospital revenues for crooked healthcare conglomerates!",
        "left-news",
        "December 5, 2023"
    ),
    (
        "SHOCKING SCAM: Solar panels proven to drain the sun's energy faster causing rapid global cooling",
        "Controversial independent climate scientists have proved that massive industrial solar panel fields absorb the electromagnetic field of the sun at accelerated rates, which will cause the earth to plunge into an artificial ice age within the next three years! Mainstream green energy lobbies are paying billions to censor this study!",
        "News",
        "January 24, 2024"
    ),
    (
        "CAUGHT ON TAPE: Media anchors rehearsing identical fabricated propaganda scripts before live broadcast",
        "Leaked satellite feeds show news anchors across dozens of major television affiliates reading word-for-word identical psychological script lines handed down directly from secret intelligence liaisons! Total proof that corporate news is 100% orchestrated theater!",
        "politics",
        "March 14, 2024"
    ),
    (
        "WAR BULLETIN: Secret space fleet deployed to intercept cloaked mothership near lunar orbit",
        "Military radar operators have confirmed that an unacknowledged division of the armed forces has launched hypersonic spacecraft to engage an extraterrestrial armada positioned in the blind spot of the moon! Full martial law is being prepared behind closed doors!",
        "News",
        "May 3, 2024"
    ),
    (
        "EXPLOSIVE: Bank executives caught secretly deleting paper currency records to force digital currency",
        "Whistleblower IT administrators at major international retail banks reveal that balance sheets are being systematically downgraded to manufacture an artificial liquidity crisis that will justify confiscating personal savings and forcing everyone into a centralized digital ID surveillance grid!",
        "US_News",
        "July 19, 2024"
    ),
    (
        "FINAL WARNING: Ancient forbidden text decoded predicting immediate global banking reset",
        "Archaeologists who discovered a sealed subterranean tomb in the Middle East have translated cryptographic hieroglyphs detailing the exact month and day that modern fiat currencies will be completely wiped out by financial elites! Buy gold and non-perishable rations immediately before it's too late!",
        "News",
        "September 1, 2024"
    )
]

def generate_expanded_dataset(target_samples_per_class=600):
    """
    Expands base article templates into a rich, diverse dataset of 1,200+ samples
    with natural lexical variations, contextual sentences, and authentic topic distributions.
    """
    random.seed(42)
    
    true_rows = []
    fake_rows = []

    # True variations
    true_prefixes = [
        "In a scheduled press briefing on Capitol Hill, ",
        "According to official statements released by regulatory authorities, ",
        "Following extensive bipartisan deliberations, ",
        "In an updated analytical report published this morning, ",
        "Representatives from international economic delegations confirmed that ",
        "During a symposium on institutional transparency, ",
        "Financial market data released by governmental agencies indicated that ",
        "In formal legislative proceedings documented by public records, "
    ]
    
    true_suffixes = [
        " Independent analysts projected that the policy will maintain fiscal stability across target sectors.",
        " Both regulatory agencies and civil stakeholders expressed cautious optimism regarding long-term implementation.",
        " The measure is slated for formal administrative review following the statutory sixty-day public comment period.",
        " Public records confirm that compliance audits will be conducted on an annual schedule.",
        " Macroeconomic indicators continue to align with consensus projections from leading academic institutions.",
        " Government spokespersons reiterated their commitment to regulatory due process and institutional integrity."
    ]

    for i in range(num_samples_per_class):
        title, text, subj, dt = random.choice(TRUE_TEMPLATES)
        intro = random.choice(true_intros)
        outro = random.choice(true_outros)
        
        full_title = title if i < len(TRUE_TEMPLATES) else f"{title} (Update {i+1})"
        full_text = f"{intro}{text}{outro}"
        
        # Introduce occasional ambiguous wording for realistic ML boundary
        if i % 25 == 0:
            full_text += " Critics raised sharp questions regarding potential partisan motivations."
        
        true_rows.append({
            "title": full_title,
            "text": full_text,
            "subject": subj,
            "date": dt
        })

    # Fake variations
    fake_prefixes = [
        "BREAKING EXCLUSIVE: You won't hear this anywhere on fake news networks! ",
        "URGENT ALERT: Uncensored whistleblower leaks have just blown the lid off the entire operation! ",
        "MUST SHARE BEFORE DELETED: Secret insider memos have fallen into our hands! ",
        "TOTAL PANIC IN WASHINGTON: The deep state is scrambling to contain this massive scandal! ",
        "SHOCKING BOMBSHELL: Hidden audio recordings confirm what patriots have suspected for years! ",
        "THE TRUTH EXPOSED: Corrupt globalists caught red-handed in treasonous coverup! ",
        "WAKE UP AMERICA: New leaked documents show the terrifying plan underway right now! "
    ]
    
    fake_suffixes = [
        " Share this urgent bulletin with every single patriot you know before Big Tech censors this website forever!",
        " The corrupt mainstream media is completely silent because they are complicit in the coverup!",
        " Whistleblowers fear for their lives as emergency tribunals prepare midnight arrest warrants!",
        " Stock up on emergency food, silver, and survival supplies right now before the impending blackout!",
        " The truth cannot be stopped and patriotic citizens are demanding immediate military intervention!",
        " This explosive scandal is the final nail in the coffin for the globalist establishment!"
    ]

    for i in range(target_samples_per_class):
        base_title, base_text, subject, date = random.choice(FAKE_ARTICLES)
        prefix = random.choice(fake_prefixes)
        suffix = random.choice(fake_suffixes)
        
        title_mod = base_title if i < len(FAKE_ARTICLES) else f"[ALERT #{i+100}] {base_title}"
        text_mod = f"{prefix}{base_text}{suffix}"
        
        fake_rows.append({
            "title": title_mod,
            "text": text_mod,
            "subject": subject,
            "date": date
        })

    # Save to True.csv
    with open(TRUE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text", "subject", "date"])
        writer.writeheader()
        writer.writerows(true_rows)

    # Save to Fake.csv
    with open(FAKE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text", "subject", "date"])
        writer.writeheader()
        writer.writerows(fake_rows)

    print(f"[Dataset Generator] Successfully generated {len(true_rows)} Real articles in {TRUE_CSV_PATH}")
    print(f"[Dataset Generator] Successfully generated {len(fake_rows)} Fake articles in {FAKE_CSV_PATH}")

if __name__ == "__main__":
    generate_expanded_dataset(target_samples_per_class=750)
