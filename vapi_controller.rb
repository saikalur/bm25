class VapiWebhooksController < ApplicationController
    skip_before_action :verify_authenticity_token
  
    LOG_FILE = Rails.root.join("log", "call_logs.json")
    DEBUG_LOG = Rails.root.join("log", "vapi_debug.log")
  
    def webhook
      unless request.post?
        return head :method_not_allowed
      end
  
      logger = Logger.new(DEBUG_LOG)
      logger.info "=== Received webhook request ==="
      logger.info "Headers: #{request.headers.env.select { |k,_| k.start_with?('HTTP_') }}"
  
      begin
        payload = JSON.parse(request.body.read)
        p "-------payload---------"
        p payload['message']['type']
        p "---------------"
        event_type = payload.dig('message', 'type') || payload['type']
        p "********************************************************"
        logger.info "Event type: #{event_type}"
        p "********************************************************"
        
        log_entry = {
          timestamp: Time.now.in_time_zone("America/Chicago").strftime('%Y-%m-%d %I:%M:%S %p %Z'),
          event_type: event_type,
          data: payload,
          ip: request.remote_ip
        }
  
        call_logs = load_logs
        call_logs << log_entry
        save_logs(call_logs)


        # Store call in DB if it's the end-of-call-report
        if event_type == 'end-of-call-report'
          call_data = payload['message'] || {}
        
          logInfo = AiFreePhoneCallLog.create!(
            phone_number: call_data['customer']['number'],
            call_type: 'Inbound',
            recording_url: call_data['recordingUrl'],
            status: call_data['status'] || 'completed',
            req_host: request.host
          )

          begin

            twillio_account_sid, twillio_auth_token, twillio_number = ""
            #Get Twillio Config
            config_details = SignUpContract.all
            config_details.each do |myconfig|
              case myconfig&.name
              when 'twillio_account_sid' then twillio_account_sid = myconfig&.content
              when 'twillio_auth_token' then twillio_auth_token = myconfig&.content
              when 'twillio_number' then twillio_number = myconfig&.content
              end
            end
            # Set Configuration
            Twilio.configure do |config|
              config.account_sid = twillio_account_sid
              config.auth_token = twillio_auth_token
            end

            message = "Thank you for talking with me. Please fill in your details here: #{ENV['HOST']}/inbound_call/#{logInfo.id}/detail\n\nCheers,\nBakerMatcher"
           
            # Send Offer Message
            TwilioMessenger.new(message, call_data['customer']['number'], twillio_number).call
            logInfo.update!(inbound_info_message_at: DateTime.now)
          rescue => e
            logger.error "Twillo error message: #{e.message}"
          end
          logger.info "Stored end-of-call-report in ai_free_phone_call_logs for call_id=#{call_data['callId']}"
        end
  
        if event_type == 'call.received'
          render json: {
            assistantId: "c2eb2fbf-23fa-4a9c-a6cc-0fcae52b3faa"
          }
        else
          render json: { status: "processed" }
        end
      rescue => e
        logger.error "Webhook error: #{e.message}"
        render json: { error: e.message }, status: :internal_server_error
      end
    end
  
    def logs
      render json: load_logs
    end
  
    def logs_view
      @logs = load_logs.reverse
      render layout: false
    end
  
    def status
      logs = load_logs
      render json: {
        status: "running",
        total_calls: logs.length,
        last_call: logs.last,
        server_time: Time.now.in_time_zone("America/Chicago").to_s
      }
    end
  
    private
  
    def load_logs
      File.exist?(LOG_FILE) ? JSON.parse(File.read(LOG_FILE), symbolize_names: true) : []
    rescue => e
      Logger.new(DEBUG_LOG).error("Error loading logs: #{e.message}")
      []
    end
  
    def save_logs(logs)
      File.write(LOG_FILE, JSON.pretty_generate(logs))
    rescue => e
      Logger.new(DEBUG_LOG).error("Error saving logs: #{e.message}")
    end
  end
  
